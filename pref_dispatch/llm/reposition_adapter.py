"""Adapter: an LLM-authored ``reposition_scores`` as a crash-safe, measured scorer.

Phase 3 evolves the region scorer the same way Phase 2 evolves the combiner, so it
needs the same two things the combiner adapter provides and the raw compiled
function does not:

* **A guard.** :mod:`pref_dispatch.reposition` calls the scorer directly inside the
  per-step driver loop. An unguarded exception on ONE idle driver therefore kills a
  whole one-hour rollout -- and with it the candidate's entire row, which the
  group-relative fitness then reads as a missing cell rather than as a bad program.
  :class:`GuardedScorer` catches per-driver and PARKS that car: it scores only the
  driver's current region, which :mod:`pref_dispatch.reposition` resolves to its
  stay rule. A program that raises everywhere therefore behaves exactly like
  repositioning switched off, and the delta fitness (which measures the gain over
  exactly that baseline) scores it 0 without needing a penalty coefficient.
  It used to return ``{}`` instead, which the kernel reads as "defer to the
  demand-gravity heuristic" -- so a broken program silently inherited a working
  policy's decisions, and ``fallback_penalty`` existed to charge for that borrowed
  credit. The penalty now defaults to 0.
* **Telemetry.** Failing silently would make a program that never runs look
  exactly like a program that runs well. The guard counts calls, failures, and the
  FIRST cause, so the evolution loop can do what Phase 2 does: one targeted repair
  attempt with the real error text, then elimination.

Two return values are deliberately NOT the same thing:

* ``{}`` from a scorer that ran fine is a **defer** -- a legitimate "I have no
  opinion about this driver, use the heuristic". Counted separately, never penalised.
* A raise, a non-dict, or a dict whose every entry is unusable is a **fallback** --
  the program is broken on this input, and the car stays put.

Training only. The deployed path (:func:`pref_dispatch.llm.basis.load_repositioner`)
calls the compiled scorer directly, so none of this affects benchmark numbers.

The capture buffer mirrors ``LLMCombiner.fleet_pick_fractions`` but is keyed by
REGION INDEX instead of skill name: it answers "does this scorer send cars
somewhere different when the objective changes?". Region indices are comparable
across the objectives being probed because the sample comes from one episode with
one fixed region layout.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from pref_dispatch.global_stats import EpisodeStats, GlobalStats
from pref_dispatch.llm.sandbox import CompiledRepositioner
from pref_dispatch.reposition import RegionState

RewardFn = Callable[[Dict], float]

# Sentinel region key for "this driver got no usable score" (defer or fallback).
# Kept in the pick distribution rather than dropped so the fractions still sum to
# 1 and a scorer that always defers reads as one stable, objective-blind policy.
NO_PICK = -1


class GuardedScorer:
    """Wrap a :class:`CompiledRepositioner` as a crash-safe, measured scorer.

    Call signature is the one :mod:`pref_dispatch.reposition` expects::

        scorer(driver_obs, phi_ep, phi_step, kappa, w) -> {region_idx: score}

    so an instance can be dropped straight into ``Repositioner(scores_fn=...)``.

    Parameters
    ----------
    scorer :
        The validated compiled repositioner.
    n_regions :
        Optional hard cap on region indices. When ``None`` (default) the bound is
        read per call from ``phi_ep.region_centres`` / the driver's
        ``relocation_points``, which is what the env actually accepts.
    """

    def __init__(
        self,
        scorer: CompiledRepositioner,
        *,
        n_regions: Optional[int] = None,
    ):
        self.scorer = scorer
        self.n_regions = int(n_regions) if n_regions is not None else None
        # Reliability telemetry (same three fields the combiner adapter exposes,
        # so the Phase-2 repair/eliminate rule transfers unchanged).
        self.n_calls = 0
        self.n_fallbacks = 0
        self.n_defers = 0
        self.first_fallback_reason: Optional[str] = None
        # Bounded (driver_obs, phi_ep, phi_step, kappa) capture for the objective
        # probe. kappa is SNAPSHOTTED because reposition.py mutates it in place as
        # drivers are assigned -- holding the live object would replay every probe
        # against whatever state the step happened to end in.
        self._capture_max = 0
        self._obs_samples: List[Tuple[Dict, EpisodeStats, GlobalStats, RegionState]] = []

    # -- internals ------------------------------------------------------- #
    def _bound(self, driver_obs: Dict, phi_ep: EpisodeStats) -> int:
        if self.n_regions is not None:
            return self.n_regions
        n = len(getattr(phi_ep, "region_centres", ()) or ())
        if not n:
            n = len(driver_obs.get("relocation_points", ()) or ())
        return int(n)

    def _note_fallback(self, reason: str) -> None:
        """Keep the FIRST fallback cause (truncated); later ones repeat it."""
        if self.first_fallback_reason is None:
            self.first_fallback_reason = reason[:300]

    def _clean(
        self,
        driver_obs: Dict,
        phi_ep: EpisodeStats,
        phi_step: GlobalStats,
        kappa: RegionState,
        w: Optional[RewardFn],
    ) -> Tuple[Dict[int, float], bool]:
        """``(scores, broke)``. ``broke`` marks a genuine program failure, so an
        honest empty return (a defer) is not charged as one."""
        n = self._bound(driver_obs, phi_ep)
        try:
            raw = self.scorer.reposition_scores(driver_obs, phi_ep, phi_step, kappa, w)
        except Exception as e:  # noqa: BLE001 -- one bad driver must not kill the hour
            self._note_fallback(f"{type(e).__name__}: {e}")
            return {}, True
        if raw is None or (isinstance(raw, dict) and not raw):
            return {}, False           # honest defer -> demand-gravity heuristic
        if not isinstance(raw, dict):
            self._note_fallback(
                f"reposition_scores returned {type(raw).__name__}, not a dict")
            return {}, True

        out: Dict[int, float] = {}
        for k, v in raw.items():
            if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
                continue
            ik = int(k)
            if not (0 <= ik < n):
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float, np.floating,
                                                         np.integer)):
                continue
            fv = float(v)
            if fv != fv or fv in (float("inf"), float("-inf")):
                continue
            out[ik] = fv
        if not out:
            self._note_fallback(
                "no usable region score survived: keys="
                f"{[repr(k) for k in list(raw)[:6]]} n_regions={n}")
            return {}, True
        return out, False

    # -- the scores_fn contract ------------------------------------------ #
    def __call__(
        self,
        driver_obs: Dict,
        phi_ep: EpisodeStats,
        phi_step: GlobalStats,
        kappa: RegionState,
        w: Optional[RewardFn] = None,
    ) -> Dict[int, float]:
        self.n_calls += 1
        if self._capture_max and len(self._obs_samples) < self._capture_max:
            self._obs_samples.append(
                (driver_obs, phi_ep, phi_step, _snapshot(kappa))
            )
        scores, broke = self._clean(driver_obs, phi_ep, phi_step, kappa, w)
        if broke:
            self.n_fallbacks += 1
            return self._stay(driver_obs)
        if not scores:
            self.n_defers += 1
        return scores

    @staticmethod
    def _stay(driver_obs: Dict) -> Dict[int, float]:
        """What a CRASH means: leave this car where it is.

        Scoring only the driver's current region makes
        :func:`pref_dispatch.reposition.reposition_targets` skip every other
        candidate, land on ``best_r == current_region`` and hit its stay rule, so
        the car does not move. A program that raises on every driver therefore
        behaves exactly like repositioning switched OFF -- which is worth exactly
        0.0 under the delta fitness, because OFF is the fitness baseline. The
        failure prices itself and no penalty coefficient is needed.

        An empty dict cannot express this: ``{}`` is the DEFER signal and falls
        through to the built-in demand-gravity heuristic, so a broken program
        would silently inherit the heuristic's decisions and its score. That
        borrowed credit is exactly what ``fallback_penalty`` used to buy back.

        Degenerate case: a driver with no region at all (``current_region < 0``)
        has nothing to stay in, so this defers. The env always assigns a region,
        so this is unreachable in practice.
        """
        try:
            cur = int(driver_obs["self"]["current_region"])
        except (KeyError, TypeError, ValueError):
            return {}
        return {cur: 1.0} if cur >= 0 else {}

    # -- reporting -------------------------------------------------------- #
    @property
    def fallback_rate(self) -> float:
        """Fraction of driver decisions on which the program actually broke."""
        return self.n_fallbacks / self.n_calls if self.n_calls else 0.0

    @property
    def defer_rate(self) -> float:
        """Fraction of driver decisions handed back to the heuristic on purpose."""
        return self.n_defers / self.n_calls if self.n_calls else 0.0

    def reset_telemetry(self) -> None:
        self.n_calls = 0
        self.n_fallbacks = 0
        self.n_defers = 0
        self.first_fallback_reason = None

    # -- objective probe --------------------------------------------------- #
    def enable_capture(self, max_samples: int = 400) -> None:
        """Record up to ``max_samples`` decision contexts so the objective probe
        can replay them under other ``w`` without extra rollouts. Clears any
        previous sample."""
        self._capture_max = int(max_samples)
        self._obs_samples = []

    def argmax_region(
        self,
        driver_obs: Dict,
        phi_ep: EpisodeStats,
        phi_step: GlobalStats,
        kappa: RegionState,
        w: Optional[RewardFn],
    ) -> int:
        """The region this scorer would send one driver to under ``w``, or
        :data:`NO_PICK` when it defers / breaks. Pure -- no telemetry, so a probe
        cannot pollute the reliability numbers."""
        n_calls, n_fb, n_df, why = (
            self.n_calls, self.n_fallbacks, self.n_defers, self.first_fallback_reason,
        )
        try:
            scores, _ = self._clean(driver_obs, phi_ep, phi_step, kappa, w)
        finally:
            self.n_calls, self.n_fallbacks, self.n_defers = n_calls, n_fb, n_df
            self.first_fallback_reason = why
        if not scores:
            return NO_PICK
        # Ties -> lowest region index, matching choose_relocation_targets.
        return min(scores, key=lambda r: (-scores[r], r))

    def fleet_region_fractions(
        self,
        w: Optional[RewardFn],
        *,
        fairness_strength: Optional[float] = None,
    ) -> Dict[int, float]:
        """Over the captured sample, the fraction of drivers sent to each region
        under objective ``w``. The Phase-3 analogue of
        ``LLMCombiner.fleet_pick_fractions``: if this distribution does not move
        between objectives, the scorer is objective-blind. Empty without a
        capture.

        ``fairness_strength`` overrides the fairness axis on every replayed
        context, which makes the strength probe a real counterfactual -- same
        drivers, same demand, same objective, only the fairness knob moved. Both
        halves of the axis are moved: ``phi_ep.fairness_strength`` (how hard the
        budget pushes) and the per-driver multipliers in the observation (who it
        pushes), because a scorer may read either. Without the override the two
        mixes being compared would come from different episodes and the difference
        would mean nothing.
        """
        n = len(self._obs_samples)
        if not n:
            return {}
        counts: Dict[int, float] = {}
        for driver_obs, phi_ep, phi_step, kappa in self._obs_samples:
            if fairness_strength is not None:
                driver_obs = _rescale_budgets(
                    driver_obs,
                    getattr(phi_ep, "fairness_strength", 0.0),
                    fairness_strength,
                )
                phi_ep = replace(phi_ep, fairness_strength=float(fairness_strength))
            r = self.argmax_region(driver_obs, phi_ep, phi_step, _snapshot(kappa), w)
            counts[r] = counts.get(r, 0.0) + 1.0
        return {k: v / n for k, v in counts.items()}


def _rescale_budgets(driver_obs: Dict, s_old: float, s_new: float) -> Dict:
    """``driver_obs`` with its fairness multipliers moved to strength ``s_new``.

    :class:`~pref_dispatch.budget.FairnessBudget` sets ``beta = exp(-s * z)`` for a
    driver's income z-score, so moving strengths is exactly ``beta ** (s_new /
    s_old)`` -- the income ranking that produced the captured betas is recovered
    without needing the incomes themselves.

    The one case this cannot recover is a context captured at strength 0: there
    every beta is 1.0 and the ranking is genuinely gone, so those samples keep
    all-1.0 budgets and only ``phi_ep.fairness_strength`` moves for them. That
    biases the strength-blindness read-out toward "blind", never away from it,
    which is the safe direction for a report-only number.
    """
    s_old = float(s_old)
    s_new = max(0.0, float(s_new))
    old = driver_obs.get("driver_budgets")
    if not old or s_old <= 0.0:
        return driver_obs
    ratio = s_new / s_old
    new = {d: float(b) ** ratio for d, b in old.items()}
    out = dict(driver_obs)
    out["driver_budgets"] = new
    did = driver_obs.get("self", {}).get("driver_id")
    out["fairness_budget"] = new.get(did, float(driver_obs.get("fairness_budget", 1.0)))
    return out


def _snapshot(kappa: RegionState) -> RegionState:
    """A private copy of kappa, so a scorer that mutates it (it must not, but a
    buggy one may) cannot corrupt the live step or a later probe."""
    return RegionState(
        demand=np.array(kappa.demand, dtype=float, copy=True),
        supply=np.array(kappa.supply, dtype=float, copy=True),
        eff_demand=np.array(kappa.eff_demand, dtype=float, copy=True),
    )
