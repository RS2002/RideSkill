"""Paradigm B vs C: the effect-vs-compute table (§5.5 / milestone 7.7).

This is proposal 5.2's headline compute-cost table. It runs, on the SAME env and
the SAME held-out preferences:

* **B (deployed method)** -- the frozen Phase-2 combiner: a pure Python function,
  ~0 online LLM calls, run over a full episode.
* **C (online upper bound)** -- :class:`OnlineLLMController`: one LLM query per
  step, run for a small number of SAMPLED steps (C is expensive by design).

and reports, per paradigm: the scalarised multi-objective (effect, on a shared
normalisation frame so B and C are comparable) and the compute cost (online LLM
calls, mean prompt/completion tokens, mean per-step latency). The point is not
that C wins or loses on effect -- it is the *effect-vs-cost trade-off*: B buys
near-zero online cost by paying an offline evolution budget once; C pays O(steps)
LLM calls forever.

Honesty: because C is sampled to ``--c-steps`` steps while B runs the full
episode, their raw metrics are NOT on the same horizon; we therefore report C's
effect on a MATCHED B run truncated to the same step budget (``B@k``) alongside
full-episode B, so the effect comparison is apples-to-apples and the cost columns
are per-step (horizon-independent). The table prints exactly what was measured.

    python -m pref_dispatch.llm.compute_table --c-steps 6 --n-prefs 2

Requires the API key in the environment / git-ignored ``.env``.
"""

from __future__ import annotations

import argparse
import random
from typing import Dict, List, Optional, Sequence

from pref_dispatch.evaluate import DispatchController, rollout
from pref_dispatch.global_stats import EpisodeStats, GlobalStats
from pref_dispatch.llm.basis import load_basis, load_frozen_combiner
from pref_dispatch.llm.client import make_llm_client
from pref_dispatch.llm.combiner_eval import build_norm_frame, make_train_prefs
from pref_dispatch.llm.config import LLMConfig
from pref_dispatch.llm.online_eval import online_rollout
from pref_dispatch.llm.paradigm_c import OnlineLLMController, StepMeter
from pref_dispatch.generalize import scalarize
from pref_dispatch.nyc_env import make_nyc_env
from pref_dispatch.preference import Preference


def _b_rollout_truncated(env_factory, combiner, skills, pref, seed, max_steps):
    """Run paradigm-B for at most ``max_steps`` steps -> metrics (for B@k match)."""
    # Reuse the online loop's structure but with the frozen combiner and no LLM:
    # simplest is a tiny inline loop mirroring evaluate.rollout with a step cap.
    from pref_dispatch.budget import FairnessBudget
    from pref_dispatch.evaluate import _make_dist
    from pref_dispatch.matching import compute_bids
    from pref_dispatch.metrics import EpisodeMetrics

    env = env_factory()
    observations, _ = env.reset(seed=seed)
    dist = _make_dist(env)
    speed = float(getattr(getattr(env, "config", None), "vehicle_speed_kmh", 0.0) or 0.0)
    phi_ep = EpisodeStats.from_observations(observations, dist=dist, speed_kmh=speed)
    metrics = EpisodeMetrics()
    income = {did: 0.0 for did in observations}
    done = False
    steps = 0
    while not done and (max_steps is None or steps < max_steps):
        phi_step = GlobalStats.from_observations(observations, dist=phi_ep.dist)
        betas = FairnessBudget(strength=pref["fairness"]).budgets(income)
        bids, _ = compute_bids(
            observations=observations, skills=skills, combiner=combiner,
            phi_ep=phi_ep, phi_step=phi_step, budgets=betas, w=phi_ep.reward_fn,
            temperature=1.0, top_k=20,
        )
        actions = {did: {"orders": oids} for did, oids in bids.items()}
        observations, rewards, dones, info = env.step(actions)
        for did, r in rewards.items():
            income[did] = income.get(did, 0.0) + float(r)
        metrics.update(rewards, info)
        done = dones["__all__"]
        steps += 1
    return metrics.finalize(total_orders=len(env._all_orders))


def main() -> None:
    ap = argparse.ArgumentParser(description="Paradigm B vs C effect/compute table.")
    ap.add_argument("--combiner", default=None, help="frozen combiner name (default: the only one).")
    ap.add_argument("--regime", default="peak")
    ap.add_argument("--split", default="test", help="held-out split for the table.")
    ap.add_argument("--n-prefs", type=int, default=2, help="held-out prefs to average.")
    ap.add_argument("--c-steps", type=int, default=6, help="sampled steps for paradigm C.")
    ap.add_argument("--num-drivers", type=int, default=150)
    ap.add_argument("--order-limit", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=5, help="candidates/driver shown to C.")
    ap.add_argument("--max-drivers", type=int, default=20, help="drivers shown to C/step.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-evolved", action="store_true")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    cfg = LLMConfig()
    if args.model:
        cfg.model = args.model
    client = make_llm_client(cfg)  # fail-fast on missing key

    skills, cards = load_basis(include_evolved=not args.no_evolved)
    combiner_b, meta = load_frozen_combiner(args.combiner, skill_names=tuple(skills))
    print("paradigm B combiner:", meta.get("combiner_name"), "| basis:", list(skills))

    def env_factory():
        return make_nyc_env(
            seed=args.seed, regime=args.regime, split=args.split,
            num_drivers=args.num_drivers, order_limit=args.order_limit,
        )

    # Shared normalisation frame (held-out prefs), so B and C effects compare.
    prefs = make_train_prefs(n=args.n_prefs, seed=args.seed + 99)  # unseen seed
    ranges = build_norm_frame(
        skills, prefs, regimes=(args.regime,), split=args.split,
        num_drivers=args.num_drivers, order_limit=args.order_limit, seed=args.seed,
    )

    b_full, b_at_k, c_eff = [], [], []
    meter_all = StepMeter()
    for pref in prefs:
        # B (full episode)
        env = env_factory()
        m_b = rollout(env, DispatchController(combiner_b, skills=skills), pref, seed=args.seed)
        b_full.append(scalarize(m_b, pref, ranges))

        # B@k (truncated to C's horizon, for an apples-to-apples effect match)
        m_bk = _b_rollout_truncated(env_factory, combiner_b, skills, pref, args.seed, args.c_steps)
        b_at_k.append(scalarize(m_bk, pref, ranges))

        # C (online LLM, sampled steps) -- share ONE meter across prefs.
        ctrl = OnlineLLMController(
            client, cards, top_k=args.top_k, max_drivers=args.max_drivers,
            meter=meter_all,
        )
        m_c, _ = online_rollout(
            env_factory(), ctrl, skills, pref, seed=args.seed, max_steps=args.c_steps
        )
        c_eff.append(scalarize(m_c, pref, ranges))

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    s = meter_all.summary()
    print("\n=== PARADIGM B vs C (effect + compute) ===")
    print(f"regime={args.regime} split={args.split} prefs={args.n_prefs} "
          f"c_steps={args.c_steps} fleet={args.num_drivers}")
    print(f"{'paradigm':<16}{'effect(scalar)':>16}{'online LLM/ep':>16}"
          f"{'tok/step':>12}{'latency/step':>14}")
    print(f"{'B (full)':<16}{_mean(b_full):>16.3f}{'~0':>16}{'-':>12}{'-':>14}")
    print(f"{'B@k (k=%d)'%args.c_steps:<16}{_mean(b_at_k):>16.3f}{'~0':>16}{'-':>12}{'-':>14}")
    print(f"{'C (online)':<16}{_mean(c_eff):>16.3f}"
          f"{s['llm_calls']/max(args.n_prefs,1):>16.1f}"
          f"{s['mean_total_tokens']:>12.0f}{s['mean_latency_s']:>14.2f}")
    print("\ncompute detail (C):", {k: round(v, 2) for k, v in s.items()})
    print(
        "\nReading: B pays a one-off OFFLINE evolution budget, then runs at ~0 "
        "online LLM cost forever; C pays "
        f"~{s['llm_calls']/max(args.n_prefs,1):.0f} LLM calls PER EPISODE "
        f"(~{s['mean_total_tokens']:.0f} tok, {s['mean_latency_s']:.2f}s each step). "
        "Compare C's effect to B@k (same horizon)."
    )


if __name__ == "__main__":
    main()
