"""Two-layer global context: ``phi_ep`` (episode-static) + ``phi_step`` (per-step).

The final-version redesign splits the single per-step ``phi`` of the v2 stack into
two layers, because the LLM-authored skills / combiner / repositioner need two very
different kinds of global information and conflating them both hid the distinction
and risked leaking the future into a "static" descriptor:

* :class:`EpisodeStats` -- ``phi_ep`` -- the **episode-static** scenario descriptor.
  Computed ONCE at :func:`pref_dispatch.evaluate.rollout` start and never again.
  It carries the fixed scenario shape (fleet size, per-vehicle capacity, speed), the
  region layout (centres + adjacency), a leak-free static map scale, the travel-time
  closure ``dist`` (the road network is fixed for the episode), and the episode's
  objective ``w`` (an LLM-authored callable reward function; ``None`` when the loop
  runs objective-blind). It contains **no future-order information** -- the static
  scale is the mean pairwise region-centre travel time, a pure function of the map
  and layout, so a frozen policy stays reproducible and no reviewer can call it an
  oracle.

* :class:`GlobalStats` -- ``phi_step`` -- the **live per-step** state (kept under its
  historical name to minimise churn). All the cheap current-observation aggregates
  the v2 stack already computed, PLUS ``kappa`` -- the per-region demand / supply
  arrays the repositioner reasons over (lifted out of
  :mod:`pref_dispatch.reposition` so both the matcher-side and reposition-side code
  read one canonical source).

``dist`` lives on ``phi_ep`` (not a separate argument) so the skill / combiner /
repositioner call surface is ``(driver_obs, order, phi_ep, phi_step)`` etc. -- the
network is episode-static, so this is the natural home and it keeps the LLM contract
to four arguments. Everything here is a cheap aggregate over the observation dict; no
env internals are touched, so it works identically for the abstract and OSMnx
networks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

Dist = Callable[[tuple, tuple], float]


# =========================================================================== #
# phi_ep -- episode-static scenario descriptor.                               #
# =========================================================================== #
@dataclass(frozen=True)
class EpisodeStats:
    """Episode-static global context. Computed once at episode start.

    Static means: nothing here changes across the steps of one episode, and
    nothing here encodes the future order stream. The static ``scale`` is derived
    purely from the region layout + network, so a frozen policy that thresholds in
    ``scale`` units is map-independent AND leak-free.
    """

    num_drivers: int
    driver_capacity: int
    speed_kmh: float
    # Region layout, straight from the observation (env is the source of truth).
    region_centres: Tuple[Tuple[float, float], ...]
    region_neighbours: Tuple[Tuple[int, ...], ...]
    # Leak-free static map scale: mean pairwise travel time between region centres
    # (a pure function of the map + layout, computed once). Skills / combiner use it
    # to set scale-free thresholds without touching any order (future) information.
    scale: float
    # The episode's objective as an LLM-authored callable reward ``w(event)->float``.
    # ``None`` = objective-blind (legacy / neutral). Only the combiner and
    # repositioner may read it; skills never do (they stay objective-specialist).
    reward_fn: Optional[Callable[[Dict], float]] = None
    # Human-readable objective label / brief, for logging + NL explanations.
    objective_label: str = ""
    # The episode's FAIRNESS STRENGTH -- the aggressiveness of the wage-equalising
    # budget multiplier the matcher applies on top of the combiner's scores
    # (``pref["fairness"]``, floored at 0, no upper cap). Episode-static because one
    # rollout runs under one preference. Phase 3 trains the repositioner ACROSS a
    # distribution of strengths, so the scorer has to be able to SEE which one it is
    # in: at strength 0 the matcher is pure efficiency and cruising toward the
    # richest demand is unopposed, while at a large strength the poorest drivers'
    # bids are multiplied up and the useful move is to put THOSE cars near demand.
    # Averaging over an invisible axis would only breed a compromise; reading it is
    # what lets ONE frozen scorer best-respond at each strength.
    fairness_strength: float = 0.0
    # ----------------------------------------------------------------------- #
    # OD matrix over the PREVIOUS window (leak-free demand geography).         #
    # ----------------------------------------------------------------------- #
    # ``od_count[i][j]`` = share of the PREVIOUS window's orders that went from
    # region i to region j (the whole matrix sums to 1.0). ``od_out[i]`` /
    # ``od_in[j]`` are its row / column sums: the share of demand that STARTED in
    # i and the share that ENDED in j. ``od_orders`` is the raw order count of
    # that window, so a policy that wants absolute volumes can multiply back.
    #
    # Episode-static and leak-free by construction: it is measured on the window
    # BEFORE the one being played (the same source the profile encoder uses,
    # never the current hour), so nothing here is a peek at the future.
    #
    # Region indices mean exactly what they mean everywhere else -- the orders
    # are bucketed with :func:`_nearest_region` against ``region_centres``, which
    # is the same nearest-centre rule as ``RidePoolEnv.region_of_points`` (the
    # driver's ``current_region``, the order's ``origin_region`` /
    # ``destination_region``) and as ``kappa`` on ``phi_step``. Row i of the OD
    # matrix, region_demand[i], and an order with ``origin_region == i`` are all
    # the same patch of map.
    #
    # Note it is not a travel-TIME matrix: the time from anywhere to anywhere is
    # already available as ``phi_ep.dist(a, b)``, so duplicating it here would
    # only be a second, staler copy.
    od_count: Tuple[Tuple[float, ...], ...] = ()
    od_out: Tuple[float, ...] = ()
    od_in: Tuple[float, ...] = ()
    od_orders: int = 0
    # Travel-time closure over the (episode-fixed) road network, in minutes. Lives
    # here because the network does not change within an episode.
    dist: Optional[Dist] = field(default=None, repr=False)

    @staticmethod
    def from_observations(
        observations: Dict[int, Dict],
        *,
        dist: Optional[Dist] = None,
        num_drivers: Optional[int] = None,
        driver_capacity: Optional[int] = None,
        speed_kmh: float = 0.0,
        reward_fn: Optional[Callable[[Dict], float]] = None,
        objective_label: str = "",
        fairness_strength: float = 0.0,
        prev_orders: Optional[Sequence[Dict]] = None,
    ) -> "EpisodeStats":
        """Build ``phi_ep`` from the FIRST observation of an episode.

        Only the region layout + fleet shape are read (both static). The scale is
        the mean pairwise region-centre travel time under ``dist`` (falls back to a
        neutral 1.0 when there is no distance backend or fewer than two regions).

        ``prev_orders`` is the PREVIOUS window's order list (dicts with ``origin``
        / ``destination`` ``(lon, lat)``, as
        :func:`pref_dispatch.llm.encode.load_prev_window_orders` returns). It is
        turned into the OD matrix. Passing nothing leaves the OD fields empty --
        the stack degrades to what it did before rather than failing.
        """
        if not observations:
            return EpisodeStats(
                num_drivers=num_drivers or 0,
                driver_capacity=driver_capacity or 0,
                speed_kmh=speed_kmh,
                region_centres=(),
                region_neighbours=(),
                scale=1.0,
                reward_fn=reward_fn,
                objective_label=objective_label,
                fairness_strength=float(fairness_strength),
                dist=dist,
            )

        any_obs = next(iter(observations.values()))
        centres = tuple(tuple(p) for p in any_obs.get("relocation_points", ()))
        neighbours = tuple(
            tuple(int(n) for n in nb)
            for nb in any_obs.get("region_neighbours", ())
        )

        if num_drivers is None:
            num_drivers = len(observations)
        if driver_capacity is None:
            # Homogeneous fleet (scenario.py guarantees this); read one driver.
            driver_capacity = int(any_obs["self"].get("capacity", 0))

        scale = _static_scale(centres, dist)
        od_count, od_out, od_in, od_orders = od_matrix(prev_orders, centres)

        return EpisodeStats(
            num_drivers=int(num_drivers),
            driver_capacity=int(driver_capacity),
            speed_kmh=float(speed_kmh),
            region_centres=centres,
            region_neighbours=neighbours,
            scale=scale,
            reward_fn=reward_fn,
            objective_label=objective_label,
            fairness_strength=float(fairness_strength),
            od_count=od_count,
            od_out=od_out,
            od_in=od_in,
            od_orders=od_orders,
            dist=dist,
        )


def _static_scale(
    centres: Tuple[Tuple[float, float], ...], dist: Optional[Dist]
) -> float:
    """Mean pairwise travel time between region centres (leak-free map scale).

    A pure function of the map + region layout -- no order / demand (future)
    information enters, so it is a stable, reproducible unit for scale-free
    thresholds. Falls back to 1.0 when unavailable.
    """
    if dist is None or len(centres) < 2:
        return 1.0
    n = len(centres)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += dist(centres[i], centres[j])
            count += 1
    return max(total / count, 1e-6) if count else 1.0


def od_matrix(
    prev_orders: Optional[Sequence[Dict]],
    centres: Tuple[Tuple[float, float], ...],
) -> Tuple[Tuple[Tuple[float, ...], ...], Tuple[float, ...], Tuple[float, ...], int]:
    """Previous-window OD matrix over regions -> ``(od_count, od_out, od_in, n)``.

    ``od_count[i][j]`` is the SHARE of ``prev_orders`` that ran from region i to
    region j (the full matrix sums to 1.0); ``od_out`` / ``od_in`` are its row /
    column sums; ``n`` is the raw order count so absolute volumes are recoverable.

    Both endpoints are bucketed with :func:`_nearest_region`, i.e. the identical
    nearest-centre rule as ``RidePoolEnv.region_of_points`` and as ``kappa`` --
    that is what makes "region i" mean the same patch of map in the OD matrix, in
    ``phi_step.region_demand``, and in an order's ``origin_region``.

    Returns empty structures when there is no region layout or no previous window
    (the profile encoder degrades the same way when a split lacks that hour).
    """
    if not centres:
        return (), (), (), 0
    r = len(centres)
    zeros_row = tuple(0.0 for _ in range(r))
    if not prev_orders:
        return tuple(zeros_row for _ in range(r)), zeros_row, zeros_row, 0

    centres_arr = np.asarray(centres, dtype=float)
    origins = _nearest_region(
        np.asarray([o["origin"] for o in prev_orders], dtype=float), centres_arr
    )
    dests = _nearest_region(
        np.asarray([o["destination"] for o in prev_orders], dtype=float), centres_arr
    )
    n = len(prev_orders)
    mat = np.zeros((r, r), dtype=float)
    np.add.at(mat, (origins, dests), 1.0 / n)

    return (
        tuple(tuple(row) for row in mat.tolist()),
        tuple(mat.sum(axis=1).tolist()),
        tuple(mat.sum(axis=0).tolist()),
        int(n),
    )


# =========================================================================== #
# phi_step -- live per-step state (historical name kept), now carrying kappa.  #
# =========================================================================== #
@dataclass(frozen=True)
class GlobalStats:
    """Per-step live global context (``phi_step``). Cheap, read-only aggregates.

    Kept under the historical ``GlobalStats`` name so existing call sites and the
    v2 stack read unchanged; the redesign only ADDS the ``kappa`` region arrays and
    the region layout needed to interpret them.
    """

    time: float
    num_pending: int
    num_drivers: int
    num_idle: int
    total_free_capacity: int
    # supply/demand imbalance: pending orders per unit of free capacity. High =>
    # demand heavy (scarce supply), low => supply heavy.
    demand_pressure: float
    # Mean solo (direct) trip time over the CURRENT pending pool -- the live scale.
    # (The episode-static counterpart is ``EpisodeStats.scale``.)
    mean_solo_time: float
    # kappa -- per-region live demand / supply, the repositioner's context. Indexed
    # by region (same order as ``EpisodeStats.region_centres``). Empty when there
    # is no region layout in the observation.
    region_demand: Tuple[float, ...] = ()
    region_supply: Tuple[float, ...] = ()

    @staticmethod
    def from_observations(
        observations: Dict[int, Dict], dist: Optional[Dist] = None
    ) -> "GlobalStats":
        if not observations:
            return GlobalStats(0.0, 0, 0, 0, 0, 0.0, 1.0, (), ())

        any_obs = next(iter(observations.values()))
        pending = any_obs["pending_orders"]
        num_pending = len(pending)
        time = float(any_obs["time"])

        num_drivers = len(observations)
        num_idle = 0
        total_free = 0
        for obs in observations.values():
            s = obs["self"]
            if s["status"] == "idle":
                num_idle += 1
            total_free += max(0, s["capacity"] - s["committed_passengers"])

        # +1 avoids div-by-zero and damps the ratio when the fleet is tiny.
        demand_pressure = num_pending / (total_free + 1.0)

        # Live scale estimate: mean direct trip time over CURRENT pending orders.
        mean_solo = 1.0
        if dist is not None and pending:
            solos = [dist(o["origin"], o["destination"]) for o in pending]
            mean_solo = sum(solos) / len(solos) if solos else 1.0

        # kappa: bucket current pending demand + free supply to region centres.
        centres = any_obs.get("relocation_points", ())
        demand, supply = _region_kappa(observations, centres)

        return GlobalStats(
            time=time,
            num_pending=num_pending,
            num_drivers=num_drivers,
            num_idle=num_idle,
            total_free_capacity=total_free,
            demand_pressure=demand_pressure,
            mean_solo_time=max(mean_solo, 1e-6),
            region_demand=demand,
            region_supply=supply,
        )


def _nearest_region(points: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Index of the nearest region centre for each point (``[N]`` -> ``[N]``).

    The same nearest-centre partition the env uses for ``current_region`` and the
    repositioner uses for demand/supply -- kept here so ``kappa`` matches both.
    """
    if points.size == 0 or centres.size == 0:
        return np.empty((0,), dtype=int)
    diff = points[:, None, :] - centres[None, :, :]
    d2 = (diff * diff).sum(axis=2)
    return d2.argmin(axis=1)


def _region_kappa(
    observations: Dict[int, Dict],
    centres,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Per-region ``(demand, supply)`` arrays for the current step (kappa).

    * demand: pending-order party sizes bucketed to the nearest region centre.
    * supply: idle / relocating drivers bucketed to the nearest region centre
      (busy cars carrying / en route to passengers are excluded -- only the free
      supply competing for the same demand).

    Returns two tuples of length ``len(centres)`` (empty when no region layout).
    """
    if not centres:
        return (), ()
    centres_arr = np.asarray(centres, dtype=float)
    r = centres_arr.shape[0]

    any_obs = next(iter(observations.values()))
    pending = any_obs.get("pending_orders", [])
    demand = np.zeros(r, dtype=float)
    if pending:
        origins = np.asarray([o["origin"] for o in pending], dtype=float)
        parties = np.asarray(
            [o.get("num_passengers", 1) for o in pending], dtype=float
        )
        np.add.at(demand, _nearest_region(origins, centres_arr), parties)

    all_drivers = any_obs.get("all_drivers", {})
    free_locs = [
        d["location"]
        for d in all_drivers.values()
        if d.get("status") in ("idle", "relocating")
    ]
    supply = np.zeros(r, dtype=float)
    if free_locs:
        np.add.at(
            supply, _nearest_region(np.asarray(free_locs, dtype=float), centres_arr), 1.0
        )

    return tuple(demand.tolist()), tuple(supply.tolist())
