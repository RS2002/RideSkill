"""Offline verification for the pref_dispatch -> benchmark integration.

Three checks, no LLM calls (paradigm B is frozen Python):

1. UNIT: PrefDispatchAdapter.act returns {did: {"orders": [...]}} with no order
   assigned to two drivers (conflict-free), on a tiny real-nyc env.
2. RECORDER GINI: a hand-fed EpisodeRecorder's summary income_gini in [0,1] and
   equals pref_dispatch.metrics.gini on the same per-driver rewards.
3. SAME-SOURCE CONSISTENCY (the key one): on the same (regime, split, seed), the
   SAME frozen controller run through the OLD path (pref_dispatch.rollout +
   EpisodeMetrics) and the NEW path (adapter + run_episode + EpisodeRecorder)
   yields matching income_gini and matching served/completed counts -- proving
   the adapter did not change execution semantics.

Run:  python -m pref_dispatch._verify_bench_adapter
"""

from __future__ import annotations

import numpy as np

from benchmark.recorder import EpisodeRecorder
from benchmark.runner import run_episode
from pref_dispatch.bench_adapter import PrefDispatchAdapter, make_pref_factory
from pref_dispatch.evaluate import DispatchController, _make_dist, rollout
from pref_dispatch.llm.basis import load_basis, load_frozen_combiner
from pref_dispatch.matching import DEFAULT_TOP_K
from pref_dispatch.metrics import gini
from pref_dispatch.nyc_env import make_nyc_env
from pref_dispatch.preference import Preference


REGIME, SPLIT, SEED = "peak", "test", 0
FLEET, ORDER_LIMIT = 40, 200  # tiny, for a fast offline check

# Which frozen combiner both paths use. ``None`` = whichever one is on disk. The
# name used to be hard-coded, which meant this file broke every time a version
# froze a differently-named champion (it was pinned to a v6 name that no longer
# exists). What check [3] actually needs is only that BOTH paths load the SAME
# program -- and passing the same ``None`` to both gives exactly that.
COMBINER = None


def _controller():
    skills, _ = load_basis(include_evolved=True)
    combiner, _ = load_frozen_combiner(COMBINER, skill_names=tuple(skills))
    # top_k comes from the shared constant, NOT a literal: check [3] compares
    # this hand-built controller against one the benchmark factory builds, and
    # the two only agree if they score the same number of orders per driver.
    return DispatchController(combiner, skills=skills, top_k=DEFAULT_TOP_K), skills


def check_unit_act() -> None:
    ctrl, _ = _controller()
    env = make_nyc_env(seed=SEED, regime=REGIME, split=SPLIT,
                       num_drivers=FLEET, order_limit=ORDER_LIMIT)
    obs, _ = env.reset(seed=SEED)
    pref = Preference({"revenue": 0.5, "service": 0.5, "fairness": 0.0})
    adapter = PrefDispatchAdapter(ctrl, pref, _make_dist(env))

    actions = adapter.act(obs)
    assert isinstance(actions, dict) and actions, "act returned empty/non-dict"
    seen = set()
    for did, a in actions.items():
        assert set(a.keys()) == {"orders"}, f"bad action shape for {did}: {a}"
        for oid in a["orders"]:
            assert oid not in seen, f"order {oid} assigned to two drivers"
            seen.add(oid)
    print(f"[1] UNIT act OK: {len(actions)} drivers, {len(seen)} conflict-free orders")


def check_recorder_gini() -> None:
    rec = EpisodeRecorder(algorithm="fake")
    # Hand-feed per-driver cumulative rewards through the same accumulator path.
    rewards_by_step = [{0: 3.0, 1: 1.0, 2: 0.0}, {0: 2.0, 1: 0.0, 2: 1.0}]

    class _Ev(dict):
        pass

    def _fake_info(rewards):
        events = {
            did: {
                "assigned_orders": [], "completed_orders": [],
                "picked_up_orders": [], "distance_moved": 0.0,
                "is_empty_move": False, "is_idle_wait": True,
            }
            for did in rewards
        }
        return {"time": 0.0, "events": events}

    class _FakeEnv:
        class _D:
            onboard_passengers = 0
        _pending_ids = []
        orders = {}
        drivers = {0: _D(), 1: _D(), 2: _D()}

        class network:  # unused by these steps
            pass

    env = _FakeEnv()
    for r in rewards_by_step:
        rec.record_step(env, _fake_info(r), rewards=r)
    rec.finalize(env)

    totals = np.array([3.0 + 2.0, 1.0 + 0.0, 0.0 + 1.0])
    expect = gini(totals)
    got = rec.summary["income_gini"]
    assert 0.0 <= got <= 1.0, f"gini out of range: {got}"
    assert abs(got - expect) < 1e-9, f"gini mismatch: {got} vs {expect}"
    print(f"[2] RECORDER gini OK: {got:.4f} == direct gini {expect:.4f}")


def check_same_source() -> None:
    pref = Preference({"revenue": 0.5, "service": 0.5, "fairness": 0.0})

    # OLD path: rollout + EpisodeMetrics.
    ctrl, _ = _controller()
    env_old = make_nyc_env(seed=SEED, regime=REGIME, split=SPLIT,
                           num_drivers=FLEET, order_limit=ORDER_LIMIT)
    m_old = rollout(env_old, ctrl, pref, seed=SEED)

    # NEW path: adapter + run_episode + EpisodeRecorder, same config.
    from benchmark.config import BenchmarkConfig
    from pref_dispatch.nyc_env import TIME_OF_DAY, _window_for_hour

    cfg = BenchmarkConfig(
        network_kind="nyc", num_drivers=FLEET, driver_capacity=4,
        nyc_splits_dir=None,
        nyc_order_path=_window_for_hour(TIME_OF_DAY[REGIME], SPLIT),
        nyc_order_limit=ORDER_LIMIT, nyc_split=SPLIT, seed=SEED,
    )
    factory = make_pref_factory(pref, combiner_name=COMBINER)
    summary, _ = run_episode(factory(cfg), algorithm_name="pref", cfg=cfg,
                             verbose=False)

    g_old, g_new = m_old["income_gini"], summary["income_gini"]
    c_old, c_new = int(m_old["completed"]), int(summary["completed"])
    print(f"[3] SAME-SOURCE: gini old={g_old:.4f} new={g_new:.4f} | "
          f"completed old={c_old} new={c_new}")
    assert abs(g_old - g_new) < 1e-6, f"gini diverged: {g_old} vs {g_new}"
    assert c_old == c_new, f"completed diverged: {c_old} vs {c_new}"
    print("[3] SAME-SOURCE OK: adapter preserves execution semantics")


def main() -> None:
    check_unit_act()
    check_recorder_gini()
    check_same_source()
    print("\nALL OK")


if __name__ == "__main__":
    main()
