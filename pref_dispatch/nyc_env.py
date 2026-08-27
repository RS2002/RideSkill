"""Real Manhattan environment builder for the preference-dispatch experiments.

The final experiments must NOT run on a toy synthetic field. This module is the
single place that turns the repo's *real* assets -- the cached Manhattan OSM
road network (``data/nyc/manhattan.gpickle``) and the preprocessed FHVHV order
windows (``data/nyc/splits/``) -- into a :class:`ride_gym.RidePoolEnv` that the
existing ``pref_dispatch`` stack consumes unchanged (it only ever touches
``env.network.shortest_path(...).travel_time``, observation coordinates, and
``env._all_orders``, all network-agnostic).

It delegates the actual wiring to the benchmark's ``make_benchmark_env`` so we
reuse the exact, already-tested real-data loader (network kind ``"nyc"``,
shared snap/matrix caches, real OD endpoints snapped onto real nodes) rather
than re-deriving it.

Time-of-day regimes are REAL demand, not a synthesized order count: each regime
names an hour window whose order volume is whatever actually occurred in the
FHVHV logs for that hour (see ``TIME_OF_DAY``). This is the "main difference is
order count across time-of-day" lever, sourced from data instead of a knob.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Dict, Optional

from ride_gym import RidePoolEnv

from benchmark.config import BenchmarkConfig, make_benchmark_env

# Repo-relative locations of the real assets (produced by
# ride_gym.data_tools.nyc.*; see that subpackage's module docstrings).
SPLITS_DIR = os.path.join("data", "nyc", "splits")
MANIFEST = os.path.join(SPLITS_DIR, "manifest.json")

# Time-of-day demand regimes, keyed to the HOUR-of-day of a real FHVHV window.
# The order volume is whatever the logs recorded for that hour (roughly: off-peak
# ~3.5k, shoulder ~6.4k, peak ~11k over the 60-min horizon), so the regimes cross
# the fleet's supply-excess -> scarcity phase transition using real demand.
TIME_OF_DAY: Dict[str, str] = {
    "offpeak": "09",
    "shoulder": "12",
    "peak": "18",
}


# Hours whose real FHVHV volume is materially above the rest of the day. Measured
# on the train split (mean orders per 60-min window): hours 08-16 all land in
# 7190-8518, while 17/18/19 are 10548/11761/11571. So the day is really TWO
# volume levels, not three -- which is why the old three-``regime`` grid
# (09/12/18) only ever offered two: 09 (7698) and 12 (7266) are the same scene.
HIGH_VOLUME_HOURS: tuple = ("17", "18", "19")


def list_windows(split: str = "train") -> list:
    """Every real order window in ``split`` as ``{file, path, start, hour, dow}``.

    The split manifest holds one entry per real 60-min FHVHV window (train has
    84 = 7 days x hours 08..19, each hour appearing exactly 7 times). Sampling
    uniformly over these entries therefore reproduces the true within-day
    frequency for free -- no hand-set regime labels, no synthesized order count.
    """
    import datetime as _dt

    with open(MANIFEST) as f:
        manifest = json.load(f)
    out = []
    for e in manifest["splits"].get(split, []):
        start = e["start"]
        d = _dt.date(int(start[0:4]), int(start[5:7]), int(start[8:10]))
        out.append({
            "file": e["file"],
            "path": os.path.join(SPLITS_DIR, e["file"]),
            "start": start,
            "hour": start[11:13],
            "dow": d.weekday(),
        })
    if not out:
        raise ValueError(f"split {split!r} has no windows in {MANIFEST}")
    return out


def _window_for_hour(hour: str, split: str) -> str:
    """Path to the (first) split window whose start hour matches ``hour``."""
    with open(MANIFEST) as f:
        manifest = json.load(f)
    entries = manifest["splits"].get(split, [])
    for e in entries:
        if e["start"][11:13] == hour:
            return os.path.join(SPLITS_DIR, e["file"])
    raise ValueError(
        f"no window starting at hour {hour!r} in split {split!r} "
        f"({MANIFEST}); available hours: "
        f"{sorted({e['start'][11:13] for e in entries})}"
    )


# --------------------------------------------------------------------------- #
# Previous-window lookup (the leak-free demand prior behind the OD matrix).    #
# --------------------------------------------------------------------------- #
def prev_window_file(window_file: str, split: str, back: int = 1) -> Optional[str]:
    """Split-relative file of the window ``back`` hours BEFORE ``window_file``.

    Resolution order, all strictly in the past so nothing here can leak the hour
    being played:

    1. the same calendar day, ``back`` hours earlier;
    2. failing that (the split only covers hours 08..19, so the 08:00 window has
       no 07:00 predecessor), the SAME hour on the closest earlier day in the
       split -- same-hour-yesterday, still measured before the episode;
    3. failing that too (a single-day split's earliest hour), ``None``: the
       caller degrades to an empty OD matrix rather than substituting anything.

    Stays inside ``split`` on purpose: a test episode never reads a train window,
    even though both would be "the past".
    """
    windows = list_windows(split)
    by_file = {w["file"]: w for w in windows}
    here = by_file.get(window_file)
    if here is None:
        return None

    day, hour = here["start"][:10], int(here["hour"])
    want = f"{hour - back:02d}"
    for w in windows:                                   # 1. same day, earlier hour
        if w["start"][:10] == day and w["hour"] == want:
            return w["file"]

    earlier = [                                         # 2. same hour, earlier day
        w for w in windows
        if w["hour"] == here["hour"] and w["start"][:10] < day
    ]
    if earlier:
        return max(earlier, key=lambda w: w["start"][:10])["file"]
    return None                                         # 3. nothing in the past


@lru_cache(maxsize=32)
def load_window_orders(path: str) -> tuple:
    """Every order in one window parquet as ``(origin, destination, party)`` dicts.

    Cached by path: the OD matrix is rebuilt once per episode and a training run
    replays the same handful of windows thousands of times. Returns a tuple (not a
    list) so the cached value cannot be mutated by a caller. Missing file -> ``()``.
    """
    import pandas as pd  # local import: heavy, and only the profiling path needs it

    if not os.path.exists(path):
        return ()
    df = pd.read_parquet(path)
    return tuple(
        {
            "origin": (float(r.origin_x), float(r.origin_y)),
            "destination": (float(r.dest_x), float(r.dest_y)),
            "num_passengers": int(r.num_passengers),
        }
        for r in df.itertuples(index=False)
    )


def prev_window_orders(window_file: str, split: str, back: int = 1) -> tuple:
    """Orders of the window ``back`` hours before ``window_file`` (``()`` if none)."""
    prev = prev_window_file(window_file, split, back=back)
    if prev is None:
        return ()
    return load_window_orders(os.path.join(SPLITS_DIR, prev))


def make_nyc_env(
    seed: int = 0,
    regime: str = "shoulder",
    split: str = "test",
    num_drivers: int = 1000,
    driver_capacity: int = 4,
    order_limit: Optional[int] = None,
) -> RidePoolEnv:
    """Build a real-Manhattan :class:`RidePoolEnv` for one time-of-day regime.

    ``regime`` selects a real hour window (:data:`TIME_OF_DAY`) on ``split``
    (``"test"`` is held out and deterministic, the right default for evaluation).
    The env replays THAT window's real orders on the real road network. Fleet
    size and capacity mirror the benchmark defaults (1000 drivers, capacity 4).
    ``order_limit`` caps orders for quick smoke runs (``None`` = the full hour).
    """
    if regime not in TIME_OF_DAY:
        raise ValueError(
            f"unknown regime {regime!r}; choose from {sorted(TIME_OF_DAY)}"
        )
    window_path = _window_for_hour(TIME_OF_DAY[regime], split)

    cfg = BenchmarkConfig(
        network_kind="nyc",
        num_drivers=num_drivers,
        driver_capacity=driver_capacity,
        # Single-file deterministic replay of exactly this hour window (bypasses
        # the multi-window random-draw sampler so a regime is reproducible).
        nyc_splits_dir=None,
        nyc_order_path=window_path,
        nyc_order_limit=order_limit,
        seed=seed,
    )
    return stamp_prev_window(make_benchmark_env(cfg),
                             os.path.relpath(window_path, SPLITS_DIR), split)


def stamp_prev_window(env: RidePoolEnv, window_file: str, split: str) -> RidePoolEnv:
    """Attach the PREVIOUS window's orders to ``env`` as ``prev_window_orders``.

    :func:`pref_dispatch.evaluate.rollout` reads this attribute to build the OD
    matrix on ``phi_ep`` without every caller having to thread the window through.
    Envs built by other paths (the abstract network, hand-made test envs) simply
    do not carry it and get an empty OD matrix.

    The stamped orders are the ones from :func:`prev_window_file` -- always an
    earlier hour, never the one about to be replayed.
    """
    try:
        env.prev_window_orders = prev_window_orders(window_file, split)
    except Exception:                  # a missing/odd manifest must not kill a run
        env.prev_window_orders = ()
    return env


def _make_env(seed: int = 0) -> RidePoolEnv:
    """Default env for the M1 sweeps: the real shoulder-hour Manhattan window."""
    return make_nyc_env(seed=seed, regime="shoulder")
