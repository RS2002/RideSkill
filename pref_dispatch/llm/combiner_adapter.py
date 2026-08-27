"""Adapter: an LLM-authored ``skill_scores`` function as a matcher-ready Combiner.

Phase 2 (§5.1) freezes the lower skills and evolves the UPPER combiner. The LLM
writes one function (final-version two-layer signature)::

    skill_scores(driver_obs, phi_ep, phi_step, w) -> {skill_name: score}

for each driver it scores every frozen skill; the platform picks the argmax skill
as that driver's scorer this step. ``w`` is the episode objective (a callable
reward fn, or ``None``); the combiner MAY call it to self-derive. This adapter is
the thin layer that turns that contract into the weight dict the existing matcher
already consumes:

    scores -> argmax over KNOWN frozen skills -> one-hot {best: 1.0}

so Phase 1 and Phase 2 share the same matcher untouched (a one-hot is just the
degenerate case of the blend the matcher already does, §5.1). With ``blend_k > 1``
the top-k skills split the weight instead, and the matcher standardizes them onto
a shared scale before mixing (see :mod:`pref_dispatch.matching`); ``blend_k = 1``
is the one-hot above, unchanged. The chosen (top) skill's name is also the
driver's **class** -- interpretable by construction, replacing the handwritten
``classify`` of :class:`~pref_dispatch.combiner.HeuristicCombiner`.

Robustness: the wrapped function is sandbox-validated, but at match time we still
defend against a score dict that (this particular driver state) is empty or names
no known skill. A guard is unavoidable -- one bad driver would otherwise kill a
whole one-hour rollout, and with it the candidate's entire row, which the
group-relative fitness reads as a missing cell rather than as a bad program.

What the guard FALLS BACK TO is the design decision. It returns an EQUAL BLEND
over the whole frozen library: the same weight for every skill, for every driver,
under every objective. That is the "no choice was made" policy
(:class:`~pref_dispatch.combiner.EqualBlendCombiner`), and it is exactly what
Phase-2 fitness subtracts as its baseline -- so a program that raises on every
driver scores exactly 0.0, the same as not having a combiner at all. The failure
prices itself and no penalty coefficient is needed.

It used to fall back to ``{skill_names[0]: 1.0}`` -- a WORKING single-skill
policy. A broken program silently inherited that skill's result, and
``fallback_penalty`` existed to charge back the borrowed credit. The penalty now
defaults to 0.

Two return values are deliberately NOT the same thing:

* ``{}`` from a scorer that ran fine is a **defer** -- a legitimate "no opinion
  about this driver". Counted separately, never treated as unreliability.
* A raise, a non-dict, or a dict naming no known skill is a **fallback** -- the
  program is broken on this input. Counted, and its FIRST cause is kept so the
  evolution loop can make one targeted repair attempt with the real error text
  before eliminating it.

Both produce the equal blend; only the second is a fault.

Training only, in the sense that matters for reported numbers: the deployed path
compiles and calls the combiner directly through this same class, but a frozen
champion has a fallback rate of 0 by construction (a program that still falls
back is eliminated during evolution, never frozen).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

from pref_dispatch.combiner import Combiner
from pref_dispatch.global_stats import EpisodeStats, GlobalStats
from pref_dispatch.llm.sandbox import CompiledCombiner
from pref_dispatch.matching import DEFAULT_BLEND_K

RewardFn = Callable[[Dict], float]

# Sentinel skill key for "this driver got no usable score" (defer or fallback).
# Kept in the pick distribution rather than mapped onto a real skill so the
# blindness probe cannot read a broken program as "always picks skill_names[0]",
# and so the fractions still sum to 1.
NO_PICK = "__no_pick__"


class LLMCombiner(Combiner):
    """Wrap a compiled ``skill_scores`` as a hard-select (argmax) combiner.

    Parameters
    ----------
    scorer :
        The validated :class:`~pref_dispatch.llm.sandbox.CompiledCombiner`.
    skill_names :
        The frozen basis' skill names. The argmax is taken ONLY over these, so a
        hallucinated key cannot select a non-existent skill.
    default_skill :
        LEGACY, kept only so old call sites and checkpoints still construct.
        Nothing selects it any more: an unusable score dict now falls back to the
        EQUAL BLEND over all skills (see :meth:`_equal_blend`), not to one
        working skill. Defaults to the first name in ``skill_names``.
    soft :
        If ``True``, blend via softmax over the scores instead of argmax one-hot
        (the §5.6 soft-mix ablation). Default ``False`` = hard argmax select.
        Equivalent to ``blend_k = 0``.
    blend_k :
        How many skills share a driver's decision (v6 item 8). ``1`` (default) is
        the hard argmax one-hot -- byte-identical to the pre-v6 path. ``k > 1``
        keeps the k highest-scored skills and splits the weight between them;
        ``<= 0`` keeps every scored skill. Before the split the selected scores
        are rescaled to ``0..1`` by their own min/max, so the weights depend only
        on how the LLM RANKED and SEPARATED the skills, never on the arbitrary
        magnitude it happened to write (one program returning 0..1 and another
        returning 0..1000 must blend the same way).
    temperature :
        Softmax temperature for the blend (ignored at ``blend_k = 1``).
    """

    def __init__(
        self,
        scorer: CompiledCombiner,
        skill_names: Sequence[str],
        *,
        default_skill: Optional[str] = None,
        soft: bool = False,
        temperature: float = 1.0,
        blend_k: int = DEFAULT_BLEND_K,
    ):
        self.scorer = scorer
        self.skill_names = tuple(skill_names)
        if not self.skill_names:
            raise ValueError("LLMCombiner needs at least one frozen skill name.")
        self.default_skill = default_skill or self.skill_names[0]
        self.soft = soft
        self.blend_k = 0 if soft else int(blend_k)
        self.temperature = temperature
        # Reliability telemetry: how many weights_for calls BROKE (a raise, a
        # non-dict, or a dict naming no known skill) and how many the program
        # declined on purpose. Fitness reads the first so an unreliable combiner
        # is measured honestly; the second is reported and never charged.
        self.n_calls = 0
        self.n_fallbacks = 0
        self.n_defers = 0
        # WHY the first fallback happened, in one line. A fallback rate alone is
        # not actionable -- v6 turns any fallback into a targeted repair attempt
        # and then elimination, and the repair prompt needs the actual cause
        # ("KeyError: 'idle_min'") rather than "0.03 of your decisions failed".
        # Only the FIRST is kept: it is bounded, and a per-driver crash in one
        # program is the same crash a few thousand times.
        self.first_fallback_reason: Optional[str] = None
        # Optional obs capture (§5.4 continuity check). When enabled, weights_for
        # records a bounded sample of the (driver_obs, phi_ep, phi_step) triples it
        # actually saw, so the fitness can measure -- WITHOUT extra env rollouts --
        # how the fleet's argmax-skill distribution shifts across a fine OBJECTIVE
        # grid (a set of w callables). A single frozen combiner under hard argmax is
        # piecewise-constant per driver; the HEADLINE (smooth, retrain-free
        # objective adaptation) needs the *fleet* mix to move gradually, which only
        # happens if drivers cross at different objective points. This buffer is
        # what lets us score that.
        self._capture_max = 0
        self._obs_samples: list = []

    # -- internal -------------------------------------------------------- #
    def _known_scores(
        self,
        driver_obs: Dict,
        phi_ep: EpisodeStats,
        phi_step: GlobalStats,
        w: Optional[RewardFn],
    ) -> Tuple[Dict[str, float], bool]:
        """Call the LLM scorer defensively; keep only finite scores for known
        skills. Returns ``(scores, broke)``.

        ``broke`` marks a genuine program failure -- a raise, a non-dict, or a
        non-empty dict none of whose entries name a known skill with a finite
        score. An HONEST empty dict (the program ran and said "no opinion here")
        is ``broke=False``: both end up on the equal blend, but only the first is
        charged as unreliability and shown to the repair prompt.

        Records the first unusable call's cause in ``first_fallback_reason`` so a
        repair attempt can be told what actually broke."""
        try:
            raw = self.scorer.skill_scores(driver_obs, phi_ep, phi_step, w)
        except Exception as e:  # noqa: BLE001 -- a per-driver crash must not kill the rollout
            self._note_fallback(f"{type(e).__name__}: {e}")
            return {}, True
        if raw is None or (isinstance(raw, dict) and not raw):
            return {}, False                 # honest defer -> the equal blend
        if not isinstance(raw, dict):
            self._note_fallback(
                f"skill_scores returned {type(raw).__name__}, not a dict")
            return {}, True
        out: Dict[str, float] = {}
        for name in self.skill_names:
            if name in raw:
                v = raw[name]
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    fv = float(v)
                    if fv == fv and fv not in (float("inf"), float("-inf")):
                        out[name] = fv
        if not out:
            self._note_fallback(
                "no finite score for any known skill; returned keys="
                f"{sorted(raw)[:8]} vs skills={list(self.skill_names)}")
            return {}, True
        return out, False

    def _note_fallback(self, reason: str) -> None:
        """Keep the FIRST fallback cause (truncated); later ones are the same."""
        if self.first_fallback_reason is None:
            self.first_fallback_reason = reason[:300]

    # -- Combiner contract ---------------------------------------------- #
    def weights_for(self, driver_obs, phi_ep, phi_step, w=None):
        self.n_calls += 1
        if self._capture_max and len(self._obs_samples) < self._capture_max:
            self._obs_samples.append((driver_obs, phi_ep, phi_step))
        scores, broke = self._known_scores(driver_obs, phi_ep, phi_step, w)
        if broke:
            self.n_fallbacks += 1
            return self._equal_blend()
        if not scores:
            self.n_defers += 1
            return self._equal_blend()

        if self.blend_k == 1:
            best = max(scores, key=scores.get)
            return {best: 1.0}
        return self._blend_weights(scores)

    def _equal_blend(self) -> Dict[str, float]:
        """What a CRASH (or an honest "no opinion") means: make NO choice.

        Every frozen skill gets the same weight, so
        :func:`~pref_dispatch.matching._active_skills` picks the same fixed slice
        of the library for every driver, every step, under every objective -- the
        same policy :class:`~pref_dispatch.combiner.EqualBlendCombiner` runs as
        the fitness BASELINE. A combiner that raises on every driver therefore IS
        the baseline and scores exactly 0.0: the failure prices itself, and no
        penalty coefficient is needed.

        It used to return ``{self.default_skill: 1.0}`` instead. That is a
        WORKING single-skill policy -- one of the frozen basis skills, deployed
        alone -- so a program that never ran silently inherited that skill's
        result, and ``fallback_penalty`` existed to charge back the borrowed
        credit. Which skill it borrowed was an accident of ``skill_names[0]``,
        and one of the frozen skills alone beats the trained combiner on 18 of
        the 30 report cells, so the accident was worth real points.
        """
        return {name: 1.0 for name in self.skill_names}

    def _blend_weights(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Softmax weights over the top ``blend_k`` scored skills.

        The selected scores are first mapped onto ``0..1`` by their own min/max,
        so the spread the softmax sees is a *relative* one: the winner's share
        depends on how clearly the LLM separated the skills, not on whether it
        wrote scores in units of 1 or 1000. A tie (zero spread) comes out as an
        equal split.
        """
        import numpy as np

        names = sorted(scores, key=lambda n: (-scores[n], n))
        if self.blend_k > 1:
            names = names[: self.blend_k]
        z = np.array([scores[n] for n in names], dtype=float)
        span = float(z.max() - z.min())
        z = (z - float(z.min())) / span if span > 1e-12 else np.zeros_like(z)
        e = np.exp(z / max(self.temperature, 1e-6))
        weights = e / e.sum()
        return {n: float(wi) for n, wi in zip(names, weights)}

    def classify(self, driver_obs, phi_ep, phi_step):
        # The driver's class IS the argmax skill name -- interpretable by design.
        # Uses w=None: classify() is a logging hook that (unlike weights_for) is
        # not handed the live objective, so we report the objective-agnostic pick.
        # The real, objective-conditioned choice is made in weights_for.
        scores, _ = self._known_scores(driver_obs, phi_ep, phi_step, None)
        if not scores:
            return NO_PICK
        return max(scores, key=scores.get)

    @property
    def fallback_rate(self) -> float:
        """Fraction of weight decisions on which the program actually broke."""
        return self.n_fallbacks / self.n_calls if self.n_calls else 0.0

    @property
    def defer_rate(self) -> float:
        """Fraction of weight decisions it declined to make on purpose (an honest
        empty dict). Same behaviour as a break -- the equal blend -- but not a
        reliability fault, so it is reported and never charged."""
        return self.n_defers / self.n_calls if self.n_calls else 0.0

    def reset_telemetry(self) -> None:
        self.n_calls = 0
        self.n_fallbacks = 0
        self.n_defers = 0
        self.first_fallback_reason = None

    # -- continuity capture (§5.4) -------------------------------------- #
    def enable_capture(self, max_samples: int = 400) -> None:
        """Start recording up to ``max_samples`` (driver_obs, phi_ep, phi_step)
        triples so the fitness can probe the fleet's argmax distribution on an
        objective grid without paying for extra env rollouts. Clears any prior
        sample."""
        self._capture_max = int(max_samples)
        self._obs_samples = []

    def argmax_pick(self, driver_obs, phi_ep, phi_step, w) -> str:
        """The skill this combiner would select for one driver under objective
        ``w``, or :data:`NO_PICK` when it defers / breaks. Pure -- no telemetry,
        so a probe cannot pollute the reliability numbers."""
        why = self.first_fallback_reason
        try:
            scores, _ = self._known_scores(driver_obs, phi_ep, phi_step, w)
        finally:
            self.first_fallback_reason = why
        if not scores:
            return NO_PICK
        return max(scores, key=scores.get)

    def fleet_pick_fractions(self, w) -> Dict[str, float]:
        """Over the captured driver sample, the fraction picking each skill under
        objective ``w``. This is the fleet-level policy whose *smooth* movement
        across objectives is the paper's headline. Empty if no capture was
        taken."""
        n = len(self._obs_samples)
        if not n:
            return {}
        counts: Dict[str, float] = {}
        for driver_obs, phi_ep, phi_step in self._obs_samples:
            pick = self.argmax_pick(driver_obs, phi_ep, phi_step, w)
            counts[pick] = counts.get(pick, 0.0) + 1.0
        return {k: v / n for k, v in counts.items()}
