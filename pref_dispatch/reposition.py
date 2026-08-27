"""Idle-driver repositioning: neighbour-region demand-gravity + sequential kappa.

This is the dispatch-layer counterpart to :mod:`pref_dispatch.budget` (fairness)
and :mod:`pref_dispatch.wage`: a small, pure, deterministic module that the
:class:`~pref_dispatch.evaluate.DispatchController` calls ONLY when repositioning
is switched on. The environment already ships a complete relocation subsystem
(the ``{"relocate": region_index}`` action, a ``RELOCATING`` status, movement
toward the target, and per-region adjacency in the observation) -- the dispatch
stack simply never used it: an idle, unmatched driver was emitted ``{"orders":
[]}`` and stayed put. This module decides, for those idle drivers, WHICH preset
region to cruise toward so empty cars drift to where near-future demand is.

Final-version redesign (A4):

* **Neighbours + global top-k hottest candidates.** Each idle driver considers its
  current region, that region's immediate neighbours (the env's ``region_neighbours``
  adjacency), AND the ``params.global_top_k`` regions with the highest live
  ``eff_demand``. The neighbour-only predecessor (archived under
  ``archive/phase3_neighbour_only/``) could not reach a city-wide hotspot that was
  not adjacent -- on a 10x10=100-region grid that left most real demand out of reach,
  which is why its reposition gain was small. Admitting the global hottest regions
  is NOT an unbounded teleport: the gravity distance decay + ``min_move_time`` +
  ``accept_floor`` stay-rules still gate every candidate, so a distant hotspot only
  wins when its demand clearly beats the empty-drive cost. Set ``global_top_k=0`` to
  recover the exact neighbour-only behaviour. Per-driver cost stays O(neighbours + k).
* **Two-layer stats + kappa + w.** The optional LLM scorer signature is
  ``reposition_scores(driver_obs, phi_ep, phi_step, kappa, w) -> {region: score}``.
  ``kappa`` is the shared, mutable ``RegionState`` (per-region demand/supply lifted
  from :class:`~pref_dispatch.global_stats.GlobalStats`); ``w`` is the episode
  objective (the repositioner MAY read it, unlike skills).
* **Sequential kappa update.** Drivers are processed in sorted-id order; when a
  driver is sent to a region, that region's demand in the shared ``kappa`` is
  decayed BEFORE the next driver scores, so cars spread instead of flocking. This
  generalises the old ``eff_demand[best_r] *= spread_decay`` into an explicit
  shared-state update the LLM scorer reads on the next driver.

Everything is deterministic given the observation (drivers iterated in sorted id
order, ties broken by lowest region index, no RNG), so a rollout stays
reproducible for a fixed env seed.

``strength`` in ``[0, 1]`` is this module's own aggressiveness knob: 0 relocates
nobody (the caller never even reaches this module), and 1 relocates most
aggressively. It scales the acceptance threshold so a small strength only moves
the few most clearly worthwhile cars. It is NOT the fairness strength -- that one
lives on the :class:`~pref_dispatch.preference.Preference` handed to ``rollout``,
reaches the matcher as the wage-equalising budget, and reaches the scorer as
``phi_ep.fairness_strength``.

The coordinated-spreading and stay-rule machinery stay here (never LLM-authored),
so the env's action contract is never in model hands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from pref_dispatch.global_stats import EpisodeStats, GlobalStats

Dist = Callable[[tuple, tuple], float]
RewardFn = Callable[[Dict], float]
# An optional per-driver region scorer:
#   (driver_obs, phi_ep, phi_step, kappa, w) -> {region_idx: base_score}
# Returns {} to defer to the demand-gravity kernel.
RepositionScoresFn = Callable[
    [Dict, EpisodeStats, GlobalStats, "RegionState", Optional[RewardFn]],
    Dict[int, float],
]


@dataclass
class RegionState:
    """Shared, mutable per-region demand/supply the repositioner reasons over.

    This is ``kappa``: seeded once per step from ``phi_step`` (the live per-region
    demand/supply arrays) and then mutated IN PLACE as drivers are assigned, so a
    later driver sees the effect of earlier drivers' choices (sequential update).
    The LLM scorer receives this object and may read ``eff_demand`` / ``supply`` /
    ``demand``; only :func:`choose_relocation_targets` mutates it.
    """

    demand: np.ndarray       # raw per-region pending demand (party sizes)
    supply: np.ndarray       # free (idle/relocating) cars nearest each region
    eff_demand: np.ndarray   # demand netted of supply, decayed as cars are sent

    @staticmethod
    def from_phi_step(
        phi_step: GlobalStats, n_regions: int, supply_weight: float
    ) -> "RegionState":
        """Seed kappa from ``phi_step``'s per-region arrays (falls back to zeros)."""
        demand = _as_region_array(getattr(phi_step, "region_demand", ()), n_regions)
        supply = _as_region_array(getattr(phi_step, "region_supply", ()), n_regions)
        eff = np.maximum(0.0, demand - supply_weight * supply)
        return RegionState(demand=demand, supply=supply, eff_demand=eff)


def _as_region_array(values, n_regions: int) -> np.ndarray:
    arr = np.zeros(n_regions, dtype=float)
    if values:
        v = np.asarray(values, dtype=float)
        m = min(len(v), n_regions)
        arr[:m] = v[:m]
    return arr


@dataclass
class Repositioner:
    """Idle-driver repositioning policy: the one self-contained handle the
    dispatch layer holds.

    A :class:`~pref_dispatch.evaluate.DispatchController` owns at most one
    ``Repositioner``; ``None`` means repositioning is off (no relocate action is
    ever emitted). This keeps the feature independent of the order-scoring skill
    stack -- a region scorer (driver x region) is a different thing from an order
    scorer (driver x order), so it lives in its own type here rather than being
    bolted onto :class:`~pref_dispatch.skills.Skill`.

    * ``strength`` in ``[0, 1]`` is the aggressiveness knob (scales the acceptance
      threshold; see :func:`choose_relocation_targets`). It is this module's own
      knob and has nothing to do with the fairness strength: the latter rides on
      the ``Preference`` and reaches the scorer as ``phi_ep.fairness_strength``,
      which is what lets one frozen repositioner best-respond across fairness
      strengths.
    * ``params`` tunes the demand-gravity kernel (defaults when ``None``).
    * ``scores_fn`` is an optional LLM-authored per-region base scorer with the
      final signature ``(driver_obs, phi_ep, phi_step, kappa, w)``; ``None`` uses
      the built-in demand-gravity heuristic. The deterministic
      coordinated-spreading, stay rules, and index emission always stay in
      :func:`choose_relocation_targets` -- never in the scorer.
    """

    strength: float = 1.0
    params: Optional["RepositionParams"] = None
    scores_fn: Optional["RepositionScoresFn"] = None

    def targets(
        self,
        observations: Dict[int, Dict],
        bids: Dict[int, list],
        phi_ep: EpisodeStats,
        phi_step: GlobalStats,
        *,
        w: Optional[RewardFn] = None,
        budgets: Optional[Dict[int, float]] = None,
    ) -> Dict[int, int]:
        """Return ``{driver_id: region_index}`` for idle, unmatched drivers.

        Only drivers with an empty bid AND ``status == "idle"`` are eligible (the
        env requires IDLE status and forbids bidding + relocate in the same step),
        so no ``InvalidActionError`` can arise from the emitted actions.

        ``budgets`` is the matcher's ``{driver_id: beta}`` fairness multipliers for
        this step (see :mod:`pref_dispatch.budget`); when given they are exposed to
        the scorer per driver. ``None`` leaves the observation untouched.
        """
        idle_empty = {
            did: obs
            for did, obs in observations.items()
            if not bids.get(did) and obs["self"]["status"] == "idle"
        }
        if not idle_empty:
            return {}
        return choose_relocation_targets(
            idle_empty,
            phi_ep,
            phi_step,
            strength=self.strength,
            params=self.params,
            reposition_scores_fn=self.scores_fn,
            w=w,
            budgets=budgets,
        )


@dataclass(frozen=True)
class RepositionParams:
    """Tunable constants for the demand-gravity reposition heuristic.

    All time-like quantities are in units of the live map scale
    (``phi_step.mean_solo_time``), so thresholds are network-independent -- the
    same convention the frozen scoring skills use.
    """

    gravity_beta: float = 1.0      # travel-time discount in the demand kernel.
    spread_decay: float = 0.6      # region demand *= this after each car sent there.
    supply_weight: float = 1.0     # subtract this * idle-cars-already-there from demand.
    min_move_time: float = 0.25    # skip relocation if target is closer than this (scale units).
    min_score: float = 0.5         # a region must beat this attractiveness to be worth it.
    recency_weight: float = 0.0    # optional up-weight of longer-waiting orders (0 = off).
    global_top_k: int = 5          # admit this many global hottest (eff_demand) regions
                                   # beyond neighbours (0 = neighbour-only, the archived
                                   # behaviour). Distance decay still gates far hotspots.


def _centres(phi_ep: EpisodeStats, observations: Dict[int, Dict]) -> np.ndarray:
    """Region-centre coordinates as an ``[R, 2]`` array (empty if no regions).

    Prefer the episode-static ``phi_ep.region_centres`` (the canonical layout);
    fall back to reading them off an observation for callers that pass a bare
    ``phi_ep`` without centres.
    """
    if getattr(phi_ep, "region_centres", ()):
        return np.asarray(phi_ep.region_centres, dtype=float)
    if observations:
        any_obs = next(iter(observations.values()))
        pts = any_obs.get("relocation_points", ())
        if pts:
            return np.asarray(pts, dtype=float)
    return np.empty((0, 2), dtype=float)


def _candidate_regions(
    current_region: int,
    region_neighbours,
    n_regions: int,
    eff_demand,
    global_top_k: int,
) -> List[int]:
    """Candidate set = current region + immediate neighbours + global top-k hottest.

    The neighbour-only version (archived under ``archive/phase3_neighbour_only/``)
    limited an idle car to its current region and adjacent ones, so the hottest
    demand in a 10x10=100-region grid was usually out of reach. We now ALSO admit
    the ``global_top_k`` regions with the highest live ``eff_demand`` (supply-netted
    demand), letting a car cruise toward a genuine city-wide hotspot even when it is
    not adjacent. This is NOT an unbounded teleport: the caller still discounts every
    candidate by cruise time (``gravity_beta`` distance decay) and enforces the
    ``min_move_time`` / ``accept_floor`` / stay rules, so a distant hotspot only wins
    when its demand clearly beats the empty-drive cost.

    Order: current region first (so a stay-rule can fire), then neighbours, then the
    hottest global regions; duplicates removed, all in-range.
    """
    cands: List[int] = []
    seen = set()
    if 0 <= current_region < n_regions:
        cands.append(current_region)
        seen.add(current_region)
    if region_neighbours is not None and 0 <= current_region < len(region_neighbours):
        for nb in region_neighbours[current_region]:
            nb = int(nb)
            if 0 <= nb < n_regions and nb not in seen:
                cands.append(nb)
                seen.add(nb)
    # Global top-k hottest by current eff_demand (supply-netted, sequentially
    # decayed), excluding regions already in the neighbour set.
    if global_top_k > 0 and eff_demand is not None and len(eff_demand) > 0:
        order = np.argsort(eff_demand)[::-1]  # descending demand
        added = 0
        for r in order:
            r = int(r)
            if added >= global_top_k:
                break
            if 0 <= r < n_regions and r not in seen:
                # Skip flat/zero-demand regions: no point admitting a "hotspot" that
                # is not actually hot (keeps the candidate set honest when demand is
                # sparse; the neighbour set + current region are always kept).
                if float(eff_demand[r]) <= 0.0:
                    continue
                cands.append(r)
                seen.add(r)
                added += 1
    return cands


def choose_relocation_targets(
    idle_observations: Dict[int, Dict],
    phi_ep: EpisodeStats,
    phi_step: GlobalStats,
    *,
    strength: float,
    params: Optional[RepositionParams] = None,
    reposition_scores_fn: Optional[RepositionScoresFn] = None,
    w: Optional[RewardFn] = None,
    budgets: Optional[Dict[int, float]] = None,
) -> Dict[int, int]:
    """Pick a relocation region for each idle, unmatched driver (neighbour-only).

    Parameters
    ----------
    idle_observations:
        Observations of ONLY the drivers eligible to relocate (empty bid AND
        ``status == "idle"``); the caller filters. Empty -> ``{}``.
    phi_ep:
        :class:`~pref_dispatch.global_stats.EpisodeStats`; supplies ``dist`` (the
        travel-time closure) and the static region layout.
    phi_step:
        :class:`~pref_dispatch.global_stats.GlobalStats`; ``phi_step.mean_solo_time``
        is the scale that makes thresholds map-independent, and its per-region
        arrays seed ``kappa``.
    strength:
        Reposition aggressiveness in ``[0, 1]``. Scales down the acceptance
        threshold so a small strength relocates only the clearly-worthwhile cars.
        ``<= 0`` returns ``{}`` (no relocation).
    params:
        :class:`RepositionParams`; defaults used when ``None``.
    reposition_scores_fn:
        Optional LLM-evolved per-region scorer with the final signature
        ``(driver_obs, phi_ep, phi_step, kappa, w)``. When it returns a non-empty
        ``{region_idx: base_score}`` for a driver those scores REPLACE the
        demand-gravity kernel's ``eff_demand`` as the base attractiveness (spreading
        + stay rules still applied on top). ``{}`` / ``None`` -> demand-gravity.
    w:
        Episode objective (callable reward fn) forwarded to the scorer.
    budgets:
        The matcher's ``{driver_id: beta}`` fairness multipliers for this step
        (``beta > 1`` = earned less than the fleet mean so its bids are boosted).
        When given, each scorer call sees its own ``driver_obs["fairness_budget"]``
        and the fleet's ``driver_obs["driver_budgets"]``; ``None`` leaves the
        observation exactly as the env produced it. The demand-gravity heuristic
        never reads them, so the injection only happens when a scorer is attached.

    Returns
    -------
    ``{driver_id: region_index}`` for the drivers that should relocate. A driver
    absent from the result stays put (the caller leaves its empty bid untouched).
    """
    if strength <= 0.0 or not idle_observations:
        return {}
    params = params or RepositionParams()
    dist = phi_ep.dist

    centres = _centres(phi_ep, idle_observations)
    n_regions = centres.shape[0]
    if n_regions == 0 or dist is None:
        return {}

    # kappa: shared, mutable per-region state seeded from phi_step. Mutated in
    # place as drivers are assigned so a later driver sees earlier choices.
    kappa = RegionState.from_phi_step(phi_step, n_regions, params.supply_weight)

    scale = max(getattr(phi_step, "mean_solo_time", 1.0), 1e-6)
    # A larger strength lowers the bar to relocate (threshold shrinks toward 0);
    # a small strength keeps only the most attractive moves.
    accept_floor = params.min_score / max(strength, 1e-6)

    # One shallow copy per STEP (not per driver) of the matcher's fairness
    # multipliers, handed read-only to the scorer. Copied so a scorer that writes
    # into it cannot corrupt the controller's own betas; shared across the drivers
    # of this step so the cost is one dict, not one per idle car.
    budget_view = dict(budgets) if budgets else None

    targets: Dict[int, int] = {}
    for did in sorted(idle_observations):
        obs = idle_observations[did]
        s = obs["self"]
        loc = s["location"]
        current_region = int(s.get("current_region", -1))
        region_neighbours = obs.get("region_neighbours")
        if region_neighbours is None:
            region_neighbours = getattr(phi_ep, "region_neighbours", None)

        cand = _candidate_regions(
            current_region, region_neighbours, n_regions,
            kappa.eff_demand, params.global_top_k,
        )
        if not cand:
            continue

        # Base per-region attractiveness: the LLM scorer if it offers an opinion,
        # else the demand-gravity kernel over the (spreading-adjusted) eff_demand.
        base_scores: Optional[Dict[int, float]] = None
        if reposition_scores_fn is not None:
            scorer_obs = obs
            if budget_view is not None:
                # The one thing the raw observation cannot tell the scorer: how the
                # matcher is about to treat THIS car. beta > 1 means it has earned
                # below the fleet mean and its bids are boosted (so it wins orders
                # from further away -- worth cruising toward demand); beta < 1 means
                # it is being damped (parking it on a hotspot buys less). Injected
                # into a shallow copy so the env's observation is never mutated.
                scorer_obs = dict(obs)
                scorer_obs["fairness_budget"] = float(budget_view.get(did, 1.0))
                scorer_obs["driver_budgets"] = budget_view
            got = reposition_scores_fn(scorer_obs, phi_ep, phi_step, kappa, w)
            if got:
                base_scores = {int(k): float(v) for k, v in got.items()}

        best_r = -1
        best_a = -np.inf
        best_tt = 0.0
        for r in cand:
            if base_scores is not None:
                base = base_scores.get(r)
                if base is None:
                    continue
            else:
                base = float(kappa.eff_demand[r])
            tt = dist(loc, tuple(centres[r])) / scale
            a = base / (1.0 + params.gravity_beta * tt)
            # Deterministic tie-break: strictly-greater keeps the lowest index.
            if a > best_a:
                best_a, best_r, best_tt = a, r, tt

        if best_r < 0:
            continue
        # Stay rules: already here, too close to bother, or not attractive enough.
        if best_r == current_region:
            continue
        if best_tt <= params.min_move_time:
            continue
        if best_a <= accept_floor:
            continue

        targets[did] = best_r
        # Sequential kappa update: discount this region's demand so the next idle
        # car near the same hotspot is nudged to the next-best neighbour region.
        kappa.eff_demand[best_r] *= params.spread_decay

    return targets
