"""Offline check that "region i" means ONE patch of map everywhere (v10, no key).

The v10 change hands the policy an OD matrix over regions on top of the region
fields it already had. That is only usable if every array indexed by a region
index agrees on what index ``i`` refers to. This file is the proof:

* the env's vectorised rule (:meth:`ride_gym.RidePoolEnv.region_of_points`), the
  single-point wrapper behind a driver's ``current_region``, and the stats-side
  :func:`pref_dispatch.global_stats._nearest_region` (which builds ``kappa`` AND
  the OD matrix) all return the SAME index for the same coordinate;
* an order's ``origin_region`` / ``destination_region`` match that rule for both
  endpoints, on a real window;
* the OD matrix bucketed inside ``phi_ep`` equals an independent recount of the
  same orders through the env's own rule -- so ``od_out[i]``,
  ``phi_step.region_demand[i]`` and ``order["origin_region"] == i`` are the same
  place;
* the OD prior is LEAK-FREE: the stamped window starts strictly BEFORE the window
  being replayed;
* the matrix is a proper distribution (sums to 1, rows/cols consistent) and
  degrades to empty -- not to garbage -- with no region layout and with no
  previous window.

Run: ``python -m pref_dispatch._verify_region_od``
"""

from __future__ import annotations

import json
import os
import random

import numpy as np

from pref_dispatch.global_stats import EpisodeStats, GlobalStats, _nearest_region, od_matrix
from pref_dispatch.nyc_env import MANIFEST, SPLITS_DIR, prev_window_file
from pref_dispatch.scenario import Scenario, build_env


def _check(label: str, ok: bool, detail: str = "") -> None:
    if not ok:
        raise AssertionError(f"{label} FAILED {detail}")
    print(f"[region-od] {label} OK {detail}".rstrip())


def _scenario() -> Scenario:
    """A small, fast, REAL window (peak hour on the held-out split)."""
    return Scenario(num_drivers=40, driver_capacity=4, speed_kmh=30.0,
                    regime="peak", split="test", order_limit=300, seed=0,
                    pref_revenue=0.5)


# --------------------------------------------------------------------------- #
# 1. The three implementations of the partition agree.                        #
# --------------------------------------------------------------------------- #
def check_one_rule(env) -> None:
    centres = np.asarray(env.relocation_points, dtype=float)
    rng = random.Random(7)
    lo = centres.min(axis=0) - 0.02
    hi = centres.max(axis=0) + 0.02
    pts = [(rng.uniform(lo[0], hi[0]), rng.uniform(lo[1], hi[1])) for _ in range(500)]

    env_r = env.region_of_points(pts)
    stats_r = _nearest_region(np.asarray(pts, dtype=float), centres)
    single_r = [env._current_region(p) for p in pts]

    _check("env vectorised rule == stats-side rule (kappa + OD)",
           bool(np.array_equal(env_r, stats_r)),
           f"({len(pts)} random points)")
    _check("env vectorised rule == the per-driver current_region wrapper",
           list(env_r) == single_r)
    _check("every index is a real region",
           int(env_r.min()) >= 0 and int(env_r.max()) < len(centres),
           f"(0..{len(centres) - 1})")


# --------------------------------------------------------------------------- #
# 2. Order fields on a real window follow that same rule.                     #
# --------------------------------------------------------------------------- #
def check_order_fields(env, obs) -> None:
    any_obs = next(iter(obs.values()))
    pending = any_obs["pending_orders"]
    _check("the window actually produced pending orders", len(pending) > 0,
           f"({len(pending)} pending)")

    o_exp = env.region_of_points([o["origin"] for o in pending])
    d_exp = env.region_of_points([o["destination"] for o in pending])
    _check("order origin_region matches the rule for every pending order",
           all(int(o["origin_region"]) == int(e) for o, e in zip(pending, o_exp)))
    _check("order destination_region matches the rule for every pending order",
           all(int(o["destination_region"]) == int(e) for o, e in zip(pending, d_exp)))

    drv = any_obs["self"]
    _check("the driver's own current_region matches the rule",
           int(drv["current_region"]) == int(env.region_of_points([drv["location"]])[0]))


# --------------------------------------------------------------------------- #
# 3. The OD matrix is bucketed by the SAME rule (independent recount).        #
# --------------------------------------------------------------------------- #
def check_od_bucketing(env, phi_ep) -> None:
    prev = env.prev_window_orders
    _check("a previous window was found and stamped on the env", len(prev) > 0,
           f"({len(prev)} orders)")
    _check("od_orders reports that window's raw size", phi_ep.od_orders == len(prev))

    # Recount independently, through the ENV's rule rather than the stats one.
    r = len(phi_ep.region_centres)
    o = env.region_of_points([x["origin"] for x in prev])
    d = env.region_of_points([x["destination"] for x in prev])
    mine = np.zeros((r, r), dtype=float)
    np.add.at(mine, (o, d), 1.0 / len(prev))
    theirs = np.asarray(phi_ep.od_count, dtype=float)
    _check("phi_ep.od_count == an independent recount through the env's rule",
           bool(np.allclose(mine, theirs)), f"({r}x{r})")

    _check("od_out / od_in are the row / column sums",
           bool(np.allclose(np.asarray(phi_ep.od_out), theirs.sum(axis=1)))
           and bool(np.allclose(np.asarray(phi_ep.od_in), theirs.sum(axis=0))))
    _check("the matrix is a distribution (sums to 1)",
           abs(theirs.sum() - 1.0) < 1e-9, f"(sum={theirs.sum():.12f})")

    # The busiest OD pair must be a plausible, non-degenerate cell.
    i, j = np.unravel_index(int(theirs.argmax()), theirs.shape)
    _check("the busiest OD pair is a real region pair with a real share",
           0 <= i < r and 0 <= j < r and theirs[i, j] > 0,
           f"(region {i} -> {j}, {theirs[i, j] * 100:.2f}% of the previous hour)")


# --------------------------------------------------------------------------- #
# 4. kappa is indexed the same way as the OD matrix.                          #
# --------------------------------------------------------------------------- #
def check_kappa_alignment(env, obs, phi_ep) -> None:
    phi_step = GlobalStats.from_observations(obs, dist=None)
    r = len(phi_ep.region_centres)
    _check("kappa arrays are region-length, like od_out / od_in",
           len(phi_step.region_demand) == r and len(phi_step.region_supply) == r
           and len(phi_ep.od_out) == r, f"({r} regions)")

    any_obs = next(iter(obs.values()))
    pending = any_obs["pending_orders"]
    demand = np.zeros(r, dtype=float)
    for o in pending:                       # bucket by the ORDER'S OWN field
        demand[int(o["origin_region"])] += float(o.get("num_passengers", 1))
    _check("region_demand[i] counts exactly the orders whose origin_region == i",
           bool(np.allclose(demand, np.asarray(phi_step.region_demand))))


# --------------------------------------------------------------------------- #
# 5. The OD prior cannot see the hour it is played on.                        #
# --------------------------------------------------------------------------- #
def check_leak_free(sc: Scenario) -> None:
    with open(MANIFEST) as f:
        entries = json.load(f)["splits"][sc.split]
    starts = {e["file"]: e["start"] for e in entries}
    here = os.path.relpath(sc.window_path, SPLITS_DIR)
    prev = prev_window_file(here, sc.split)
    _check("the played window resolves to an earlier window",
           prev is not None and starts[prev] < starts[here],
           f"({starts[prev]} < {starts[here]})")

    # Every window in every split must resolve to something strictly earlier or
    # to nothing at all -- never to itself, never forward.
    bad = []
    for split in ("train", "val", "test"):
        with open(MANIFEST) as f:
            es = json.load(f)["splits"][split]
        st = {e["file"]: e["start"] for e in es}
        for fname in st:
            p = prev_window_file(fname, split)
            if p is not None and st[p] >= st[fname]:
                bad.append((split, fname, p))
    _check("no window in any split resolves forward or to itself", not bad,
           f"({len(bad)} violations)")


# --------------------------------------------------------------------------- #
# 6. Degradation is empty, not garbage.                                       #
# --------------------------------------------------------------------------- #
def check_degrades() -> None:
    centres = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    empty, out, inn, n = od_matrix([], centres)
    _check("no previous window -> an all-zero matrix, od_orders 0",
           n == 0 and len(empty) == 3 and all(v == 0.0 for row in empty for v in row)
           and out == (0.0, 0.0, 0.0) and inn == (0.0, 0.0, 0.0))

    nothing = od_matrix([{"origin": (0.0, 0.0), "destination": (1.0, 0.0)}], ())
    _check("no region layout -> everything empty", nothing == ((), (), (), 0))

    one = od_matrix([{"origin": (0.1, 0.0), "destination": (0.0, 0.9)}], centres)
    _check("a single order lands in the nearest-centre cell",
           one[0][0][2] == 1.0 and one[1] == (1.0, 0.0, 0.0)
           and one[2] == (0.0, 0.0, 1.0) and one[3] == 1)

    blank = EpisodeStats.from_observations({}, dist=None)
    _check("an empty observation gives empty OD fields, not a crash",
           blank.od_count == () and blank.od_orders == 0)


def main() -> None:
    sc = _scenario()
    env = build_env(sc)
    obs, _info = env.reset(seed=0)

    def dist(a, b):
        return env.network.shortest_path(a, b).travel_time

    phi_ep = EpisodeStats.from_observations(
        obs, dist=dist, speed_kmh=sc.speed_kmh,
        prev_orders=env.prev_window_orders,
    )

    check_one_rule(env)
    check_order_fields(env, obs)
    check_od_bucketing(env, phi_ep)
    check_kappa_alignment(env, obs, phi_ep)
    check_leak_free(sc)
    check_degrades()
    print("\n[region-od] ALL region/OD consistency checks passed (no API key used).")


if __name__ == "__main__":
    main()
