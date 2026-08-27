"""Reward-authoring stage (§Phase-2): translate a platform PREFERENCE into ONE
concrete, sandboxed ``reward(event) -> float`` the combiner is then composed to
maximise.

This is a deliberately SEPARATE LLM stage from combiner composition. Its whole job
is to turn a preference -- given as a natural-language brief, a metric-weight dict,
or a concrete :class:`~ride_gym.rewards.RewardFunction` instance -- into a single
frozen objective:

* **NL brief / weight dict**  -> the LLM authors a full reward body over the
  per-step ``event`` dict (Q1 = "LLM authors reward body"), gated by a
  ``reward_understanding`` chain-of-thought (it must explain the preference before
  writing the reward) and validated in the AST sandbox before use.
* **concrete RewardFunction** -> authoring is SKIPPED; the instance already IS the
  objective. We snapshot its live coefficients / source for provenance and wrap it.

Both paths converge on an :class:`AuthoredReward`: a ``(driver_id, event) -> float``
callable ready to inject as ``env.reward_function`` (via
``make_benchmark_env(reward_function=...)``), plus the metadata the combiner prompt
and the frozen artefact record as provenance. There is NO runtime preference dial:
one preference -> one reward -> one number.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

from pref_dispatch.llm.client import LLMClient


# --------------------------------------------------------------------------- #
# Stage-dependent authoring contract.                                          #
# --------------------------------------------------------------------------- #
_TRAIN_PROBE_SPEC = (
    "(TRAINING) The reward you write is the GROUND-TRUTH objective the combiner's "
    "probes must learn to read. Translate the preference faithfully -- do NOT "
    "reshape it to fit any probe. The probes adapt to this objective, not the "
    "other way round."
)

_INFER_PROBE_SPEC = (
    "(DEPLOYMENT) The combiner's probes are FROZEN. Write the reward so that the "
    "existing probe geometry can actually detect each of its terms. Prefer terms "
    "that the probe reads by contrasting event shapes on: completion (drop-off "
    "paying vs assignment-only), seat/party size (party-2 vs party-1), trip length "
    "(long vs short solo time), empty/idle aversion (move vs empty move), volume "
    "(one vs two assignments), and pickup wait (service time minus solo time). If a "
    "term cannot be read by such a contrast, express it through one that can."
)


def _probe_spec_of(stage: str) -> str:
    return _TRAIN_PROBE_SPEC if stage == "train" else _INFER_PROBE_SPEC
from pref_dispatch.llm.extract import extract_json, require_explanation
from pref_dispatch.llm.prompts.reward_author import (
    REQUIRED_EXPLANATION_FIELDS,
    REQUIRED_FIELDS,
    build_reward_prompt,
)
from pref_dispatch.llm.reward_spec import RewardLike, describe_reward, reward_spec
from pref_dispatch.llm.sandbox import (
    SandboxError,
    compile_reward,
    normalise_terms,
    validate_reward,
)
from ride_gym.rewards import RewardFunction


class RewardAuthoringError(RuntimeError):
    """Raised when no valid reward can be authored after all repair attempts."""


@dataclass
class AuthoredReward:
    """A concrete platform reward objective, whatever form the preference took.

    ``fn`` is the per-driver, per-step callable the env grades with, always in the
    env's ``(driver_id, event) -> float`` shape (the authored ``reward(event)`` is
    adapted to it). ``meta`` carries the provenance the combiner prompt and the
    frozen artefact embed: for an authored reward the CoT ``reward_understanding`` +
    ``code``; for a given instance a coefficient/source snapshot.
    """

    fn: Callable[[int, Dict], float]
    meta: Dict = field(default_factory=dict)

    @property
    def code(self) -> Optional[str]:
        return self.meta.get("code")

    @property
    def name(self) -> str:
        return self.meta.get("reward_name", "platform_reward")

    @property
    def spec_text(self) -> str:
        """The prompt-ready description handed to the combiner (§6 TARGET REWARD)."""
        return self.meta.get("spec_text", "")

    @property
    def is_authored(self) -> bool:
        """True if the LLM wrote the reward body; False for a given instance."""
        return bool(self.meta.get("authored"))


def _adapt_event_reward(reward_event_fn: Callable[[Dict], float]) -> Callable[[int, Dict], float]:
    """Wrap an authored ``reward(event)`` into the env's ``(driver_id, event)`` shape.

    The env calls ``reward_function(driver_id, event)``; the LLM authors a
    ``reward(event)`` that ignores driver identity (per the contract). This adapter
    bridges the two without leaking the driver id into the sandbox surface.
    """

    def _env_reward(_driver_id: int, event: Dict) -> float:
        return float(reward_event_fn(event))

    return _env_reward


def _check_fields(obj: Dict) -> None:
    missing = [f for f in REQUIRED_FIELDS if not str(obj.get(f, "")).strip()]
    if missing:
        raise SandboxError(f"response missing required field(s): {missing}")
    require_explanation(obj, REQUIRED_EXPLANATION_FIELDS)


# --------------------------------------------------------------------------- #
# Term-weight normalisation (2026-08-13).                                      #
# --------------------------------------------------------------------------- #
# The LLM authors the reward's coefficients freely, so an ``nl`` reward can carry
# wildly different scales across terms (e.g. detour at 4x the fare rate, a
# completion bonus of 20 next to a per-minute price of 0.5). A combiner probes w
# on synthetic events to read those term weights; absolute scale is unreadable and
# makes an extreme term look dominant even when the author meant a mild tilt.
#
# We therefore rescale the authored reward so the term weights SUM TO ONE, using
# the same synthetic-event surface the reward was validated against. The probes
# below each isolate ONE additive term (the additivity rule guarantees the total
# is the sum of per-term prices); the scale S is the sum of their absolute
# contributions, and the final fn is ``w / S`` (S=1 falls through unchanged). This
# keeps the RATIOS the author intended while making the scale comparable across
# objectives, and it is scale-invariant by construction -- the GRPO fitness and
# the equal-blend baseline both divide by the same objective scale.
#
# Only the AUTHORED (nl / weights-with-client) path is normalised here. The raw
# and key-free weights paths draw coefficients from a fixed envelope that is
# already unit-consistent, and the given-RewardFunction path is used verbatim.
_PROBE_EVENTS: Tuple[Tuple[str, Dict], ...] = (
    # (label, event) -- each event moves ONE term away from the zero base.
    ("assignment", {
        "assigned_orders": [1], "assigned_party_sizes": {1: 1},
        "assigned_solo_times": {1: 2.0}, "assigned_service_times": {1: 2.0},
        "completed_orders": [], "picked_up_orders": [], "distance_moved": 0.0,
        "time_moved": 0.0, "is_empty_move": False, "is_idle_wait": False,
        "extra_detour_time": 0.0,
    }),
    ("completion", {
        "assigned_orders": [1], "assigned_party_sizes": {1: 1},
        "assigned_solo_times": {1: 2.0}, "assigned_service_times": {1: 2.0},
        "completed_orders": [1], "picked_up_orders": [1], "distance_moved": 0.0,
        "time_moved": 0.0, "is_empty_move": False, "is_idle_wait": False,
        "extra_detour_time": 0.0,
    }),
    ("seat", {
        "assigned_orders": [1], "assigned_party_sizes": {1: 2},
        "assigned_solo_times": {1: 2.0}, "assigned_service_times": {1: 2.0},
        "completed_orders": [], "picked_up_orders": [], "distance_moved": 0.0,
        "time_moved": 0.0, "is_empty_move": False, "is_idle_wait": False,
        "extra_detour_time": 0.0,
    }),
    ("service_minute", {
        "assigned_orders": [1], "assigned_party_sizes": {1: 1},
        "assigned_solo_times": {1: 2.0}, "assigned_service_times": {1: 4.0},
        "completed_orders": [], "picked_up_orders": [], "distance_moved": 0.0,
        "time_moved": 0.0, "is_empty_move": False, "is_idle_wait": False,
        "extra_detour_time": 0.0,
    }),
    ("detour", {
        "assigned_orders": [1], "assigned_party_sizes": {1: 1},
        "assigned_solo_times": {1: 2.0}, "assigned_service_times": {1: 2.0},
        "completed_orders": [], "picked_up_orders": [], "distance_moved": 0.0,
        "time_moved": 0.0, "is_empty_move": False, "is_idle_wait": False,
        "extra_detour_time": 2.0,
    }),
    ("empty_move", {
        "assigned_orders": [], "assigned_party_sizes": {},
        "assigned_solo_times": {}, "assigned_service_times": {},
        "completed_orders": [], "picked_up_orders": [], "distance_moved": 2.0,
        "time_moved": 2.0, "is_empty_move": True, "is_idle_wait": False,
        "extra_detour_time": 0.0,
    }),
    ("idle", {
        "assigned_orders": [], "assigned_party_sizes": {},
        "assigned_solo_times": {}, "assigned_service_times": {},
        "completed_orders": [], "picked_up_orders": [], "distance_moved": 0.0,
        "time_moved": 0.0, "is_empty_move": False, "is_idle_wait": True,
        "extra_detour_time": 0.0,
    }),
)


def _normalise_reward_terms(reward_event_fn, code: str = "",
                            ) -> Tuple[Callable[[Dict], float], float]:
    """Rescale an authored ``reward(event)`` so its COEFFICIENTS sum to one.

    v2 (2026-08-26): normalise on the reward's own COEFFICIENTS, not on its
    event responses. The old path divided by the sum of ``|fn(probe)|`` over
    synthetic events, which folds the probe events' magnitudes into the scale
    (an assignment probe with one order and a detour probe with two minutes do
    not contribute equally per unit of cost). The combiner reads w by probing
    differences, so any single global scale is invisible to it -- but a
    coefficient-sourced scale is deterministic, independent of the probe grid,
    and is what "the term weights should sum to one" literally means.

    Coefficients are extracted from the source ``code`` by regex: the top-level
    ``NAME = number`` assignments that are then USED in a product that also
    touches an ``event[...]`` field (i.e. a per-order/per-step price, not an
    accumulator like ``total``). If no such coefficient is named -- e.g. the LLM
    inlined every price as a literal -- we fall back to the probe-response path
    (the old behaviour), so a structurally-free reward is still scaled rather
    than left arbitrarily large.

    Returns ``(scaled_fn, S)``: ``scaled_fn`` is ``fn / S`` (a single global
    factor, so the author's term RATIOS are preserved exactly) and ``S`` is the
    scale it was divided by (``1.0`` when the reward is left unchanged).
    """
    if code:
        coefs = _extract_named_coefficients(code)
        if coefs:
            total = sum(abs(c) for c in coefs)
            if math.isfinite(total) and total > 1e-12:
                S = float(total)

                def scaled(inputs: Dict, fn=reward_event_fn, S=S) -> float:
                    return float(fn(inputs)) / S

                return scaled, S
    # Fallback: no named coefficient, or a degenerate one -- use the probe
    # response surface (the v1 behaviour), which at least keeps the reward
    # comparable across objectives.
    from pref_dispatch.llm.sandbox import normalise_terms
    return normalise_terms(
        reward_event_fn, [ev for _l, ev in _PROBE_EVENTS], domain="reward",
    )


_COEF_ASSIGN_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(-?\d+\.?\d*)\s*(?:#.*)?$",
    re.M,
)


def _extract_named_coefficients(code: str) -> list:
    """Top-level ``NAME = number`` constants that act as per-order prices.

    A constant is a coefficient, not an accumulator or a temporary, if it appears
    in a product that also reads an ``event[...]`` field somewhere in the body --
    ``total += -WAIT_PRICE * wait`` uses ``WAIT_PRICE`` as a price, while ``total
    = 0.0`` never multiplies anything. We match assignments and then check each
    name against the product uses in the source. Best-effort: an inlined literal
    price is simply not found, and the caller falls back to the probe scale.
    """
    coefs = []
    for m in _COEF_ASSIGN_RE.finditer(code):
        name, val = m.group(1), float(m.group(2))
        # the name must be reused in a product (NAME * or * NAME) and that product
        # must also mention an event-field access, to count as a per-order price.
        if not re.search(rf"{re.escape(name)}\s*\*|\*\s*{re.escape(name)}", code):
            continue
        if not re.search(rf"event\s*\[", code):
            continue
        coefs.append(val)
    return coefs


def _build_authored(obj: Dict, preference_spec: str) -> AuthoredReward:
    """Validate an LLM response into an :class:`AuthoredReward` (authored path)."""
    _check_fields(obj)
    reward_event_fn = compile_reward(obj["code"])
    ok, why = validate_reward(reward_event_fn)
    if not ok:
        raise SandboxError(f"reward invalid: {why}")

    # Coefficient normalisation: rescale the reward so the SUM OF ITS OWN
    # COEFFICIENTS is 1, by dividing the whole function by that sum. This is a
    # single global factor -- the author's term RATIOS are preserved exactly --
    # and unlike the old event-response scale it is deterministic and independent
    # of the probe grid. The env grades with the SCALED function; the original
    # code and the scale are both kept in meta for provenance.
    scaled_fn, norm_scale = _normalise_reward_terms(reward_event_fn, obj["code"])

    name = str(obj["reward_name"]).strip()
    meta = {
        "authored": True,
        "reward_name": name,
        "objective": obj["objective"].strip(),
        "description": obj["description"].strip(),
        "reward_understanding": obj["reward_understanding"].strip(),
        "code": obj["code"].rstrip(),
        "preference_spec": preference_spec,
        "norm_scale": norm_scale,
        # What the combiner prompt shows as the TARGET REWARD it composes FOR.
        "spec_text": (
            "This platform reward was authored FROM the preference below and is now "
            "the FIXED objective you compose the fleet to maximise.\n\n"
            f"Reward understanding: {obj['reward_understanding'].strip()}\n\n"
            f"Objective: {obj['objective'].strip()}\n\n"
            f"How it scores (per driver, per step):\n{obj['description'].strip()}\n\n"
            "NOTE: the reward's term weights were NORMALISED so they sum to 1 "
            "(each term's price is its share of the total absolute weight; scale "
            "does not matter, only the ratios between terms).\n\n"
            f"```python\n{obj['code'].rstrip()}\n```"
        ),
    }
    return AuthoredReward(fn=_adapt_event_reward(scaled_fn), meta=meta)


def _ask(client: LLMClient, prompt: Dict[str, str], temperature=None) -> Dict:
    raw = client.complete(prompt["system"], prompt["user"], temperature=temperature)
    return extract_json(raw)


def author_reward(
    client: LLMClient,
    preference: RewardLike,
    *,
    n_repair: int = 2,
    temperature: float = 0.7,
    stage: str = "train",
    log: Callable[[str], None] = print,
) -> AuthoredReward:
    """Turn any supported ``preference`` FORM into one concrete reward objective.

    * A concrete :class:`RewardFunction` instance -> authoring is SKIPPED: the
      instance is the objective, wrapped with a coefficient/source snapshot.
    * A natural-language brief or weight dict     -> the LLM authors a
      ``reward(event)`` body, gated by ``reward_understanding`` and sandbox-
      validated, retried up to ``n_repair`` times on any compile/validate error.

    ``stage`` selects the authoring contract:
      * ``"train"`` (default) -- TRANSLATE the preference faithfully. The seeded
        objective is the ground truth the combiner's probes must learn to read, so
        the reward must NOT be reshaped to fit any probe; the probes adapt to it.
      * ``"infer"``  -- TRANSLATE FOR THE CURRENT PROBE. Deployment: the combiner's
        probes are frozen, so the reward is written so that the existing probe
        geometry can actually detect its terms (see ``probe_spec``).

    Never reads or writes an API key: the ``client`` already holds its own env-only
    key (see :mod:`pref_dispatch.llm.client`); this function only calls ``complete``.
    """
    # --- Given a concrete reward: skip authoring, snapshot for provenance. ---- #
    if isinstance(preference, RewardFunction):
        spec = describe_reward(preference)
        snapshot = {
            k: getattr(preference, k)
            for k in (
                "assignment_bonus", "revenue_coef", "service_time_coef",
                "detour_coef", "empty_move_penalty", "idle_penalty",
            )
            if hasattr(preference, k)
        }
        log(f"[reward] using given {type(preference).__name__} (authoring skipped)")
        meta = {
            "authored": False,
            "reward_name": type(preference).__name__,
            "objective": "maximise the platform's given reward function",
            "description": "A concrete RewardFunction instance supplied by the "
                           "platform; used verbatim as the fitness objective.",
            "reward_understanding": "Reward supplied directly; no translation needed.",
            "reward_snapshot": snapshot,
            "spec_text": spec,
        }
        # The env calls reward_function(driver_id, event) -- the instance already
        # has that exact __call__ signature, so inject it directly.
        return AuthoredReward(fn=preference, meta=meta)

    # --- NL / weights: the LLM authors the reward body (with repair loop). ---- #
    preference_spec = reward_spec(preference)
    if not preference_spec:
        raise RewardAuthoringError("empty preference: nothing to translate into a reward.")

    feedback: Optional[str] = None
    last_err = ""
    probe_spec = _probe_spec_of(stage)
    for attempt in range(n_repair + 1):
        prompt = build_reward_prompt(preference_spec,
                                     repair_feedback=feedback,
                                     stage=stage,
                                     probe_spec=probe_spec)
        try:
            obj = _ask(client, prompt, temperature=temperature)
            authored = _build_authored(obj, preference_spec)
            log(
                f"[reward] authored {authored.name!r}: "
                f"{authored.meta['objective']}"
            )
            log(f"         reward_understanding: {authored.meta['reward_understanding']}")
            return authored
        except Exception as e:  # noqa: BLE001 -- retry on any compile/validate error
            last_err = f"{type(e).__name__}: {e}"
            feedback = last_err
            log(f"[reward] attempt {attempt} failed: {last_err}")

    raise RewardAuthoringError(
        f"no valid reward after {n_repair + 1} attempts; last error: {last_err}"
    )
