"""Scenario sampling & domain randomization for the v2 generalization upgrade.

v1 evolved the skill basis and the Phase-2 combiner at a SINGLE fixed operating
point (800 drivers, capacity 4, three fixed regimes). That is exactly why the
frozen policies -- like the MARL baselines -- were never *asked* to generalize to
a different fleet size, capacity, speed, or time-of-day. This module is the v2
answer: it turns "a scenario" into a concrete :class:`BenchmarkConfig` +
:class:`Preference`, and samples scenarios across five axes so the evolution loop
sees variety and the frozen policy transfers with ZERO retraining.

The five axes (all real levers on :class:`BenchmarkConfig`, verified):

* ``num_drivers``      -- fleet size (~100..2000).
* ``driver_capacity``  -- homogeneous per-vehicle capacity (~1..10).
* ``speed_kmh``        -- driver speed; :class:`OSMnxNetwork` applies it as
  ``time = distance / speed`` (so it genuinely rescales every travel time).
* ``regime``           -- time-of-day window (:data:`TIME_OF_DAY`), i.e. a REAL
  FHVHV hour whose order volume/shape is whatever actually occurred.
* ``pref_revenue``     -- the platform preference (revenue vs service), passed to
  the dispatch loop; ``fairness`` stays 0 this round (deferred).

Critical coupling (verified in ``ride_gym/env.py`` ``_assign_orders``): the env
does NOT raise when an order's party exceeds a driver's capacity -- it greedily
drops the largest parties and leaves them pending. But an order whose party size
ALONE exceeds the whole (homogeneous) fleet's capacity can never be seated by
anyone and simply expires at timeout, systematically poisoning the metrics. So
whenever we randomize capacity we MUST clamp the order party upper bound to
``min(GLOBAL_MAX_PARTY, capacity)``. :meth:`ScenarioSampler.sample` and
:func:`build_env` both enforce this.

Homogeneous fleet assumption (this round): every driver shares one capacity and
one speed, so scalar ``driver_capacity`` / ``speed_kmh`` suffice (the env's
per-vehicle ``driver_capacities`` / ``driver_speeds`` lists are unused). Fairness
and rescheduling are deferred.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from benchmark.config import BenchmarkConfig, make_benchmark_env
from pref_dispatch.nyc_env import (
    HIGH_VOLUME_HOURS,
    MANIFEST,
    SPLITS_DIR,
    TIME_OF_DAY,
    _window_for_hour,
    list_windows,
    stamp_prev_window,
)
from pref_dispatch.preference import Preference
from ride_gym import RidePoolEnv

# The env's order stream tops out at this many passengers even when a big fleet
# capacity would allow more; keeps party sizes in the realistic ride-pool range
# and matches the ``max_party_size=4`` the MARL checkpoints trained under.
GLOBAL_MAX_PARTY: int = 4


# --------------------------------------------------------------------------- #
# Axis ranges (the domain-randomization envelope).                            #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScenarioRanges:
    """The domain-randomization envelope: inclusive ranges for each axis.

    Defaults are the v2 training envelope. The deployment/MARL anchor point
    (1000 drivers, capacity 4, speed 35, default reward) sits inside it, so the
    frozen policy is evaluated in-distribution there while every other sampled
    scenario is genuinely off the MARL training point.

    ``fleet_dist`` controls HOW fleet sizes are drawn from ``fleet``: ``"uniform"``
    (legacy) gives every fleet size equal probability; ``"loguniform"`` gives every
    ORDER OF MAGNITUDE equal mass, so small fleets (the scarcity regime the frozen
    policy historically failed at) stop being a rare tail of the draw. ``order_limits``,
    when set, makes each scenario draw its per-rollout order cap from those choices
    instead of the fixed ``order_limit`` -- covering demand-volume variety (a few
    hundred orders to a full hour) inside one training distribution.
    """

    fleet: Tuple[int, int] = (100, 2000)
    fleet_dist: str = "uniform"          # "uniform" | "loguniform"
    capacity: Tuple[int, int] = (1, 10)
    speed_kmh: Tuple[float, float] = (25.0, 45.0)
    regimes: Tuple[str, ...] = tuple(sorted(TIME_OF_DAY))  # offpeak/shoulder/peak
    revenue: Tuple[float, float] = (0.0, 1.0)
    # Optional cap on orders per rollout (None = full real hour). During
    # evolution a cap keeps rollouts cheap; the champion re-eval uses None.
    order_limit: Optional[int] = None
    # When set, each scenario draws its order cap from these choices (overrides
    # the fixed ``order_limit``); None entries mean "full hour".
    order_limits: Optional[Sequence[Optional[int]]] = None

    def as_prompt_dict(self) -> Dict[str, object]:
        """Compact human/LLM-readable summary of the envelope (for encode.py)."""
        return {
            "fleet": list(self.fleet),
            "fleet_dist": self.fleet_dist,
            "capacity": list(self.capacity),
            "speed_kmh": list(self.speed_kmh),
            "regimes": list(self.regimes),
            "revenue_weight": list(self.revenue),
            "order_limits": list(self.order_limits)
            if self.order_limits else self.order_limit,
        }


# --------------------------------------------------------------------------- #
# Simulated-clock lookup (day-of-week + hour), from the split manifest.        #
# --------------------------------------------------------------------------- #
_DOW = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def _window_start(hour: str, split: str) -> Optional[str]:
    """Absolute ``start`` timestamp string of the window used for ``hour``.

    Returns ``None`` if the manifest is unavailable (encode.py degrades). This is
    the SIMULATED clock source: the FHVHV window's real calendar timestamp, NOT
    wall-clock time.
    """
    try:
        with open(MANIFEST) as f:
            manifest = json.load(f)
    except OSError:
        return None
    for e in manifest["splits"].get(split, []):
        if e["start"][11:13] == hour:
            return e["start"]
    return None


def simulated_clock(regime: str, split: str) -> Tuple[Optional[str], Optional[int]]:
    """``(day_of_week_name, hour_int)`` of the simulated window, or (None, None).

    Parsed from the window's absolute ``start`` timestamp (``YYYY-MM-DD HH:...``)
    with no pandas dependency -- Zeller-free via :func:`datetime.date.weekday`.
    """
    hour = TIME_OF_DAY.get(regime)
    if hour is None:
        return None, None
    start = _window_start(hour, split)
    if start is None:
        return None, int(hour)
    import datetime as _dt

    d = _dt.date(int(start[0:4]), int(start[5:7]), int(start[8:10]))
    return _DOW[d.weekday()], int(start[11:13])


# --------------------------------------------------------------------------- #
# A single scenario.                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class Scenario:
    """One concrete point in the randomization space.

    ``service = 1 - pref_revenue`` and ``fairness = 0`` (deferred), so a scenario
    fully determines both the environment and the platform preference the
    dispatch loop is scored under.
    """

    num_drivers: int
    driver_capacity: int
    speed_kmh: float
    regime: str
    split: str = "train"
    order_limit: Optional[int] = None
    pref_revenue: float = 0.5
    seed: int = 0
    # v6: an explicit REAL order window (split-relative file, e.g.
    # ``train\window_0041.parquet``). When set it fully replaces the ``regime``
    # lookup -- the scene is one concrete hour that actually happened, drawn
    # uniformly from the split's 84 windows, instead of one of three hand-picked
    # hours. ``regime`` is kept only as a human label for such scenarios.
    window: Optional[str] = None

    @property
    def max_party(self) -> int:
        """Order party upper bound under this fleet's capacity (the clamp)."""
        return max(1, min(GLOBAL_MAX_PARTY, int(self.driver_capacity)))

    @property
    def preference(self) -> Preference:
        rev = float(self.pref_revenue)
        return Preference(
            {"revenue": rev, "service": 1.0 - rev, "fairness": 0.0}
        )

    @property
    def window_path(self) -> str:
        """Concrete parquet path of this scenario's order window."""
        if self.window:
            return os.path.join(SPLITS_DIR, self.window)
        return _window_for_hour(TIME_OF_DAY[self.regime], self.split)

    @property
    def clock(self) -> Tuple[Optional[str], Optional[int]]:
        """Simulated ``(day_of_week, hour)`` for this scenario's window."""
        if self.window:
            for w in list_windows(self.split):
                if w["file"] == self.window:
                    return _DOW[w["dow"]], int(w["hour"])
            return None, None
        return simulated_clock(self.regime, self.split)

    def to_config(self) -> BenchmarkConfig:
        """Concrete single-file-replay :class:`BenchmarkConfig` for this scenario."""
        return BenchmarkConfig(
            network_kind="nyc",
            num_drivers=int(self.num_drivers),
            driver_capacity=int(self.driver_capacity),
            speed_kmh=float(self.speed_kmh),
            nyc_splits_dir=None,          # deterministic single-window replay
            nyc_order_path=self.window_path,
            nyc_order_limit=self.order_limit,
            nyc_split=self.split,
            random_party_size=True,
            max_party_size=self.max_party,  # <-- the capacity coupling clamp
            seed=int(self.seed),
        )

    def label(self) -> str:
        dow, hr = self.clock
        when = f"{dow[:3]}{hr:02d}h" if dow else self.regime
        return (
            f"f{self.num_drivers}_c{self.driver_capacity}_"
            f"s{self.speed_kmh:g}_{when}_rev{self.pref_revenue:g}"
        )


def build_env(scenario: Scenario, reward_function=None) -> RidePoolEnv:
    """Build a real-Manhattan :class:`RidePoolEnv` for ``scenario``.

    Thin wrapper over :func:`make_benchmark_env` on ``scenario.to_config()`` so
    the env is byte-identical to what the benchmark harness builds from the same
    config. The party clamp is already baked into the config.

    ``reward_function`` is an optional passthrough (§Phase-2): when set, THAT
    callable grades every driver instead of the config's ``DefaultRewardFunction``,
    so the same authored objective is injected across every domain-randomised
    scenario (fleet/capacity/speed/regime vary; the reward does not). ``None``
    keeps the anchor reward -- fully backward-compatible.

    The env is stamped with the PREVIOUS window's orders
    (:func:`pref_dispatch.nyc_env.stamp_prev_window`), which is what
    :func:`pref_dispatch.evaluate.rollout` turns into the OD matrix on ``phi_ep``.
    That window is always an earlier hour than the one being replayed, so the
    prior stays leak-free.
    """
    env = make_benchmark_env(scenario.to_config(), reward_function=reward_function)
    return stamp_prev_window(
        env, os.path.relpath(scenario.window_path, SPLITS_DIR), scenario.split
    )


# --------------------------------------------------------------------------- #
# The sampler.                                                                 #
# --------------------------------------------------------------------------- #
class ScenarioSampler:
    """Draw random :class:`Scenario` s from a :class:`ScenarioRanges` envelope.

    Deterministic given the seeded :class:`random.Random`. Party sizes are always
    clamped to the sampled capacity (via :attr:`Scenario.max_party`), so no
    scenario ever strands orders that no vehicle could seat.
    """

    def __init__(
        self,
        ranges: Optional[ScenarioRanges] = None,
        rng: Optional[random.Random] = None,
        *,
        split: str = "train",
    ):
        self.ranges = ranges or ScenarioRanges()
        self.rng = rng or random.Random(0)
        self.split = split

    def sample(
        self, *, seed: Optional[int] = None,
        ranges: Optional[ScenarioRanges] = None,
    ) -> Scenario:
        r = self.rng
        rg = ranges or self.ranges
        num_drivers = self._sample_fleet(rg)
        capacity = r.randint(*rg.capacity)
        speed = r.uniform(*rg.speed_kmh)
        regime = r.choice(list(rg.regimes))
        revenue = r.uniform(*rg.revenue)
        # Each scenario carries its OWN env seed so the same Scenario reproduces
        # byte-identical orders/driver init; default to a fresh draw when unset.
        env_seed = seed if seed is not None else r.randint(0, 2**31 - 1)
        # Order volume: draw from the per-scenario choices when provided (covers
        # the small-order regime inside one distribution), else the fixed cap.
        order_limit = (
            r.choice(list(rg.order_limits)) if rg.order_limits else rg.order_limit
        )
        return Scenario(
            num_drivers=num_drivers,
            driver_capacity=capacity,
            speed_kmh=round(speed, 2),
            regime=regime,
            split=self.split,
            order_limit=order_limit,
            pref_revenue=round(revenue, 3),
            seed=env_seed,
        )

    def _sample_fleet(self, rg: ScenarioRanges) -> int:
        """Fleet size from ``rg.fleet`` under ``rg.fleet_dist``.

        ``"loguniform"`` gives every order of magnitude of scale equal probability
        mass (the scarcity regime below ~400 cars stops being a rare tail of a
        uniform draw); ``"uniform"`` is the legacy equal-probability draw."""
        lo, hi = int(rg.fleet[0]), int(rg.fleet[1])
        if lo >= hi:
            return lo
        if rg.fleet_dist == "loguniform":
            raw = math.exp(self.rng.uniform(math.log(lo), math.log(hi)))
            return max(lo, min(hi, int(round(raw))))
        return self.rng.randint(lo, hi)

    def sample_batch(self, k: int, *, base_seed: Optional[int] = None) -> List[Scenario]:
        """Draw ``k`` scenarios. When ``base_seed`` is given, each scenario's env
        seed is pinned deterministically (``base_seed + i``) so an entire batch is
        reproducible and shareable across candidates within one comparison round.
        """
        out: List[Scenario] = []
        for i in range(k):
            s = base_seed + i if base_seed is not None else None
            out.append(self.sample(seed=s))
        return out

    def sample_batch_stratified(
        self,
        k: int,
        *,
        bands: Sequence[Tuple[int, int]],
        regimes: Sequence[str],
        order_modes: Sequence[Optional[int]],
        base_seed: Optional[int] = None,
    ) -> List[Scenario]:
        """Draw ``k`` scenarios stratified across (fleet-band x regime) cells.

        Cells are visited round-robin so EVERY band x regime combination appears
        in the batch (a small fleet + a large fleet, peak + offpeak, capped +
        full-hour order volume all in ONE comparison grid). Within a cell the
        fleet is drawn log-uniformly inside the band; the order cap alternates
        capped (``order_modes``) / full hour (None). Env seeds stay pinned to
        ``base_seed + i`` -- the same reproducibility contract as
        :meth:`sample_batch` -- so every candidate of a GRPO round is rolled on
        the identical grid.
        """
        cells = [(lo, hi, reg) for (lo, hi) in bands for reg in regimes]
        base = self.ranges
        out: List[Scenario] = []
        for i in range(k):
            lo, hi, regime = cells[i % len(cells)]
            limit = (
                order_modes[i % len(order_modes)]
                if (order_modes and i % 2 == 0) else None
            )
            cell = dataclasses.replace(
                base,
                fleet=(float(lo), float(hi)),
                regimes=(regime,),
                order_limit=limit,
                order_limits=None,
            )
            out.append(self.sample(
                seed=base_seed + i if base_seed is not None else None,
                ranges=cell,
            ))
        return out


    def sample_real_windows(
        self,
        k: int,
        *,
        base_seed: Optional[int] = None,
        min_high_volume: int = 1,
        ranges: Optional[ScenarioRanges] = None,
    ) -> List[Scenario]:
        """v6: draw ``k`` scenarios on REAL order windows, each a full hour.

        The window is drawn uniformly from every window in the split (train has
        84 = 7 days x hours 08..19), so the batch's demand-volume spread is the
        real one and no ``order_limit`` is ever set. This replaces the old
        "cap the stream at 600/800/1500 orders" knob, which did not reduce a
        scene's scale -- it kept the earliest N requests and so produced a
        ~3-minute rush followed by ~57 minutes of an empty city, a scene that
        does not occur in the data.

        ``min_high_volume`` guarantees at least that many windows come from the
        genuinely busy hours (:data:`~pref_dispatch.nyc_env.HIGH_VOLUME_HOURS`,
        mean 10.5k-11.8k orders vs 7.2k-8.5k for the rest). A uniform draw puts
        3 of 12 hours in that band, so a small round can otherwise miss the busy
        regime entirely by luck -- and the busy cells are exactly where the gate
        is hardest.

        Fleet/capacity/speed/preference are drawn from the envelope as usual and
        env seeds are pinned to ``base_seed + i`` when given, so a round's grid
        is reproducible and identical for every program rolled on it.
        """
        rg = ranges or self.ranges
        windows = list_windows(self.split)
        high = [w for w in windows if w["hour"] in HIGH_VOLUME_HOURS]
        if not high:                       # split without a busy hour: degrade
            high = windows
        picks = [self.rng.choice(windows) for _ in range(k)]
        # Force the guarantee by overwriting slots that were not already busy,
        # choosing the slots from the front so the draw stays deterministic.
        n_high = sum(1 for w in picks if w["hour"] in HIGH_VOLUME_HOURS)
        need = min(max(0, int(min_high_volume) - n_high), k)
        if need:
            slots = [i for i, w in enumerate(picks)
                     if w["hour"] not in HIGH_VOLUME_HOURS][:need]
            for i in slots:
                picks[i] = self.rng.choice(high)
        out: List[Scenario] = []
        for i, w in enumerate(picks):
            sc = self.sample(
                seed=base_seed + i if base_seed is not None else None,
                ranges=dataclasses.replace(rg, order_limit=None, order_limits=None),
            )
            hour = w["hour"]
            regime = "peak" if hour in HIGH_VOLUME_HOURS else "offpeak"
            out.append(dataclasses.replace(
                sc, window=w["file"], regime=regime, order_limit=None,
            ))
        return out


def sample_scenario_set(
    n: int,
    *,
    seed: int = 0,
    ranges: Optional[ScenarioRanges] = None,
    split: str = "train",
    pin_env_seeds: bool = True,
) -> List[Scenario]:
    """Convenience: a reproducible batch of ``n`` scenarios.

    ``seed`` seeds BOTH the axis draws and (when ``pin_env_seeds``) the per-
    scenario env seeds, so calling this with the same ``seed`` reproduces the
    exact same scenario list -- the right primitive for a held-out evaluation
    set (use a different ``seed`` + ``split="test"`` than the training draw).
    """
    sampler = ScenarioSampler(ranges=ranges, rng=random.Random(seed), split=split)
    base = seed if pin_env_seeds else None
    return sampler.sample_batch(n, base_seed=base)
