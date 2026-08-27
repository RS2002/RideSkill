"""Offline verification for Feature 2 (driver-WAGE fairness). No LLM key needed.

Checks:
 1. wage definition: ``driver_wage_from_event`` = sum(solo_time x party) over
    assigned orders, and ignores every penalty term (idle/detour/empty-move) --
    a pure-idle event yields wage 0 while its shaping reward would be negative.
 2. budget consumes wage: with strength>0, a below-mean-wage driver gets beta>1
    and an above-mean-wage driver beta<1; strength=0 => all budgets exactly 1.0
    (the mechanism is a no-op, so fairness=0 runs are byte-identical to legacy).
 3. wage_gini consistency: the benchmark ``EpisodeRecorder`` and the online
    ``EpisodeMetrics`` compute the SAME wage_gini from the same per-driver wage
    stream (single-source definition agrees across both rollout paths).
 4. online rollout: a full small rollout produces finite wage_gini/wage_mean and
    the fairness budget only reorders matches (service stays within a small band
    of the fairness-off baseline -- it never discards served demand).
"""
from __future__ import annotations

import numpy as np

from benchmark.recorder import EpisodeRecorder
from pref_dispatch.budget import FairnessBudget
from pref_dispatch.evaluate import DispatchController, rollout
from pref_dispatch.llm.basis import load_basis, load_frozen_combiner
from pref_dispatch.metrics import EpisodeMetrics, gini
from pref_dispatch.preference import Preference
from pref_dispatch.scenario import Scenario, build_env
from pref_dispatch.wage import driver_wage_from_event


def test_wage_definition() -> None:
    ev = {
        "assigned_orders": [1, 2],
        "assigned_solo_times": {1: 10.0, 2: 4.0},
        "assigned_party_sizes": {1: 2, 2: 1},
        # penalties that MUST NOT enter wage:
        "is_idle_wait": True, "is_empty_move": True, "extra_detour_time": 99.0,
        "assigned_service_times": {1: 999.0, 2: 999.0},
    }
    assert driver_wage_from_event(ev) == 10.0 * 2 + 4.0 * 1, driver_wage_from_event(ev)
    assert driver_wage_from_event({"is_idle_wait": True}) == 0.0
    print(f"[1] wage-def OK: fare-only = {driver_wage_from_event(ev)} "
          f"(ignores idle/detour/service penalties); idle step -> 0.")


def test_budget_consumes_wage() -> None:
    wage = {0: 5.0, 1: 20.0, 2: 100.0}
    b = FairnessBudget(strength=0.5).budgets(wage)
    assert b[0] > 1.0 > b[2], b          # poorest boosted, richest damped
    assert b[0] > b[1] > b[2], b          # monotone in wage rank
    b0 = FairnessBudget(strength=0.0).budgets(wage)
    assert b0 == {d: 1.0 for d in wage}, b0
    print(f"[2] budget-consumes-wage OK: beta poor={b[0]:.3f} > mid={b[1]:.3f} "
          f"> rich={b[2]:.3f}; strength=0 => all 1.0 (legacy no-op).")


def test_wage_gini_consistency() -> None:
    # Same per-driver wage stream fed through both paths must give one gini.
    steps = [
        {0: {"assigned_orders": [1], "assigned_solo_times": {1: 8.0},
             "assigned_party_sizes": {1: 2}, "completed_orders": [],
             "picked_up_orders": [], "distance_moved": 0.0,
             "is_empty_move": False, "is_idle_wait": False,
             "assigned_service_times": {1: 10.0}, "extra_detour_time": 0.0},
         1: {"assigned_orders": [], "completed_orders": [], "picked_up_orders": [],
             "distance_moved": 0.0, "is_empty_move": False, "is_idle_wait": True,
             "assigned_solo_times": {}, "assigned_party_sizes": {},
             "assigned_service_times": {}, "extra_detour_time": 0.0}},
        {0: {"assigned_orders": [], "completed_orders": [], "picked_up_orders": [],
             "distance_moved": 0.0, "is_empty_move": False, "is_idle_wait": True,
             "assigned_solo_times": {}, "assigned_party_sizes": {},
             "assigned_service_times": {}, "extra_detour_time": 0.0},
         1: {"assigned_orders": [2], "assigned_solo_times": {2: 3.0},
             "assigned_party_sizes": {2: 1}, "completed_orders": [],
             "picked_up_orders": [], "distance_moved": 0.0,
             "is_empty_move": False, "is_idle_wait": False,
             "assigned_service_times": {2: 5.0}, "extra_detour_time": 0.0}},
    ]
    m = EpisodeMetrics()
    for evs in steps:
        m.update(rewards={d: 0.0 for d in evs}, info={"events": evs})
    metrics_wg = m.finalize(total_orders=2)["wage_gini"]

    rec = EpisodeRecorder(algorithm="fake")

    class _Env:  # minimal stand-ins for the recorder's env probes
        _pending_ids: list = []
        orders: dict = {}
        drivers: dict = {}
    env = _Env()
    for evs in steps:
        rec.record_step(env, {"events": evs, "time": 0.0},
                        assign_log={}, rewards={d: 0.0 for d in evs})
    wages = np.array([rec._driver_acc[d]["wage"] for d in sorted(rec._driver_acc)])
    rec_wg = gini(wages)
    assert abs(metrics_wg - rec_wg) < 1e-9, (metrics_wg, rec_wg)
    print(f"[3] wage_gini-consistency OK: EpisodeMetrics {metrics_wg:.6f} == "
          f"recorder {rec_wg:.6f} (single wage definition across both paths).")


def test_online_rollout_budget_preserves_service() -> None:
    sc = Scenario(num_drivers=80, driver_capacity=4, speed_kmh=35.0,
                  regime="offpeak", split="train", order_limit=60,
                  pref_revenue=0.5, seed=0)
    env = build_env(sc)
    skills, _ = load_basis(include_evolved=True)
    # ``None`` = whichever combiner is frozen on disk. This check is about the
    # wage budget, not about which champion is current, and a hard-coded name
    # breaks the file every time a version freezes a new one.
    comb, _m = load_frozen_combiner(None, skill_names=tuple(skills))
    ctrl = DispatchController(comb, skills=skills, top_k=20)
    base = rollout(env, ctrl,
                   Preference({"revenue": 0.5, "service": 0.5, "fairness": 0.0}),
                   seed=0)
    fair = rollout(env, ctrl,
                   Preference({"revenue": 0.5, "service": 0.5, "fairness": 1.0}),
                   seed=0)
    for r in (base, fair):
        assert np.isfinite(r["wage_gini"]) and np.isfinite(r["wage_mean"]), r
    # The budget only reorders WHO wins a contested order; service must not
    # collapse (stays within a small band of the fairness-off baseline).
    assert fair["service_rate"] >= base["service_rate"] - 0.05, (
        base["service_rate"], fair["service_rate"])
    print(f"[4] online-rollout OK: strength0 wage_gini={base['wage_gini']:.4f} "
          f"service={base['service_rate']:.3f} | strength1 "
          f"wage_gini={fair['wage_gini']:.4f} service={fair['service_rate']:.3f} "
          f"(service preserved).")


if __name__ == "__main__":
    test_wage_definition()
    test_budget_consumes_wage()
    test_wage_gini_consistency()
    test_online_rollout_budget_preserves_service()
    print("\nALL FEATURE-2 (WAGE FAIRNESS) OFFLINE CHECKS PASSED")
