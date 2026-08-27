"""Offline verification for A3 every-pair scoring (large-cap KNN). No LLM key.

The final version raises the default candidate cap so that at realistic operating
points EVERY feasible nearby (driver, order) pair is scored, while keeping the KNN
prune as the O(N*K) safety valve for city-scale fleets. This script pins that
behaviour on hand-built observations where the scored-candidate set is observable:

 (a) DEFAULT cap is large (>= 50) -- the A3 "score every feasible pair" intent.
 (b) top_k <= 0 => true all-pairs: every driver's candidate set is every pending
     order (no spatial prune at all).
 (c) large default cap (60) with fewer pending than the cap => still all-pairs
     (prune never engages below the cap).
 (d) prune IS a safety valve: with more pending than a small explicit top_k, each
     driver scores exactly top_k orders (its nearest), not all of them.
 (e) the scored survivors under a big cap are a SUPERSET of those under a tiny cap
     (raising the cap only ever adds candidates, never drops a near one).

We count candidates by instrumenting ``_knn_candidates`` directly (the single
choke point matching.py routes every pair through).
"""
from __future__ import annotations

import numpy as np

from pref_dispatch.matching import DEFAULT_TOP_K, _knn_candidates, compute_bids
from pref_dispatch.combiner import SingleSkillCombiner
from pref_dispatch.global_stats import EpisodeStats, GlobalStats
from pref_dispatch.skills import default_skill_basis


def _euclid(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _mk_obs(did, loc, pending):
    return {
        "self": {
            "driver_id": did,
            "location": loc,
            "status": "idle",
            "capacity": 4,
            "committed_passengers": 0,
            "current_region": 0,
            "assigned_order_details": [],
        },
        "all_drivers": {},
        "pending_orders": pending,
        "time": 0.0,
        "relocation_points": ((0.0, 0.0), (10.0, 10.0)),
        "region_neighbours": ((1,), (0,)),
    }


def _pending(n):
    # n orders spread along a line so nearest-K is well-defined and deterministic.
    return [
        {"order_id": i, "origin": (float(i), 0.0), "destination": (float(i) + 1.0, 0.0),
         "num_passengers": 1, "waiting_time": 0.0}
        for i in range(n)
    ]


def _obs_map(n_drivers, n_pending):
    pend = _pending(n_pending)
    obs = {d: _mk_obs(d, (float(d) * 0.5, 0.0), pend) for d in range(n_drivers)}
    all_drivers = {
        d: {"location": o["self"]["location"], "status": "idle", "onboard_passengers": 0}
        for d, o in obs.items()
    }
    for o in obs.values():
        o["all_drivers"] = all_drivers
    return obs


def test_default_cap_is_large() -> None:
    assert DEFAULT_TOP_K >= 50, DEFAULT_TOP_K
    print(f"[a] default-cap OK: DEFAULT_TOP_K={DEFAULT_TOP_K} (>=50; scores every "
          f"feasible pair at realistic operating points).")


def test_top_k_zero_is_all_pairs() -> None:
    obs = _obs_map(n_drivers=4, n_pending=30)
    near = _knn_candidates(obs, obs[0]["pending_orders"], top_k=0)
    for did, idxs in near.items():
        assert sorted(idxs) == list(range(30)), (did, len(idxs))
    print(f"[b] all-pairs OK: top_k<=0 -> every driver scores all 30 pending "
          f"orders (no spatial prune).")


def test_large_default_below_cap_is_all_pairs() -> None:
    # Fewer pending than the cap => prune never engages (all orders survive).
    obs = _obs_map(n_drivers=4, n_pending=DEFAULT_TOP_K - 5)
    near = _knn_candidates(obs, obs[0]["pending_orders"], top_k=DEFAULT_TOP_K)
    n = DEFAULT_TOP_K - 5
    for did, idxs in near.items():
        assert sorted(idxs) == list(range(n)), (did, len(idxs))
    print(f"[c] under-cap OK: {n} pending < cap {DEFAULT_TOP_K} -> all-pairs "
          f"(prune dormant).")


def test_prune_is_safety_valve() -> None:
    # More pending than a small explicit cap => each driver scores exactly K,
    # and they are its nearest (smallest |origin.x - loc.x|).
    obs = _obs_map(n_drivers=3, n_pending=30)
    K = 5
    near = _knn_candidates(obs, obs[0]["pending_orders"], top_k=K)
    pend = obs[0]["pending_orders"]
    for did, idxs in near.items():
        assert len(idxs) == K, (did, len(idxs))
        loc = obs[did]["self"]["location"]
        # the chosen K must be the K nearest by pickup distance
        d2_all = sorted(range(len(pend)),
                        key=lambda i: _euclid(pend[i]["origin"], loc))
        assert set(idxs) == set(d2_all[:K]), (did, sorted(idxs), d2_all[:K])
    print(f"[d] safety-valve OK: {len(pend)} pending, top_k={K} -> each driver "
          f"scores exactly its {K} nearest orders (O(N*K) bound holds).")


def test_raising_cap_only_adds() -> None:
    obs = _obs_map(n_drivers=3, n_pending=30)
    small = _knn_candidates(obs, obs[0]["pending_orders"], top_k=5)
    big = _knn_candidates(obs, obs[0]["pending_orders"], top_k=20)
    for did in obs:
        assert set(small[did]) <= set(big[did]), (
            did, sorted(small[did]), sorted(big[did]))
    print(f"[e] monotone OK: nearest-5 candidates are a subset of nearest-20 for "
          f"every driver (raising the cap only adds candidates).")


def test_compute_bids_end_to_end_all_pairs() -> None:
    # Sanity: compute_bids runs at top_k=0 (all-pairs) and returns conflict-free
    # bids under the two-layer signature.
    skills = {s.name: s for s in default_skill_basis()}
    comb = SingleSkillCombiner("service")
    obs = _obs_map(n_drivers=4, n_pending=12)
    phi_ep = EpisodeStats.from_observations(obs, dist=_euclid, speed_kmh=35.0)
    phi_step = GlobalStats.from_observations(obs, dist=_euclid)
    bids, classes = compute_bids(
        observations=obs, skills=skills, combiner=comb,
        phi_ep=phi_ep, phi_step=phi_step,
        budgets={d: 1.0 for d in obs}, w=None, top_k=0,
    )
    # conflict-free one-to-one: no order awarded to two drivers.
    awarded = [oid for oids in bids.values() for oid in oids]
    assert len(awarded) == len(set(awarded)), bids
    assert all(len(oids) <= 1 for oids in bids.values()), bids
    print(f"[f] compute_bids all-pairs OK: top_k=0 end-to-end, conflict-free bids "
          f"= { {d: b for d, b in bids.items() if b} }.")


if __name__ == "__main__":
    test_default_cap_is_large()
    test_top_k_zero_is_all_pairs()
    test_large_default_below_cap_is_all_pairs()
    test_prune_is_safety_valve()
    test_raising_cap_only_adds()
    test_compute_bids_end_to_end_all_pairs()
    print("\nALL MATCHING EVERY-PAIR (A3) OFFLINE CHECKS PASSED")
