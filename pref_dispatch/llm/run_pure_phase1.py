"""One-command PURE-Phase-1 evolution (proposal step 4).

Evolves ONE fleet-wide dispatch scorer under the FIXED default MARL reward
(``income_mean`` -- the exact ``DefaultRewardFunction`` the RL baselines train on).
The model must explain the reward first (chain-of-thought), then maximise it. No
self-authored fitness, no upper combiner: this is the "single policy, same
objective as the MARL agents" rung of the effectiveness ladder.

Requires the API key in the environment (never in the repo):

    export YIBU_API_KEY=sk-...
    python -m pref_dispatch.llm.run_pure_phase1 --regimes offpeak shoulder peak

Use ``--regimes peak --order-limit 400`` for a fast wiring run.
"""

from __future__ import annotations

import argparse
import random

from pref_dispatch.global_stats import GlobalStats
from pref_dispatch.llm.client import make_llm_client
from pref_dispatch.llm.config import LLMConfig
from pref_dispatch.llm.encode import encode_env_profile
from pref_dispatch.llm.evolve_pure_phase1 import (
    evolve_pure_phase1,
    freeze_pure_phase1,
)
from pref_dispatch.llm.fitness_eval import EVAL_NUM_DRIVERS, EVAL_ORDER_LIMIT
from pref_dispatch.nyc_env import make_nyc_env


def main() -> None:
    ap = argparse.ArgumentParser(description="Evolve one pure-Phase-1 scorer.")
    ap.add_argument("--regimes", nargs="+", default=["offpeak", "shoulder", "peak"])
    ap.add_argument("--profile-regime", default=None,
                    help="regime used to build the env profile (default: first).")
    ap.add_argument("--split", default="train")
    ap.add_argument("--generations", type=int, default=4)
    ap.add_argument("--lam", type=int, default=2)
    ap.add_argument("--order-limit", type=int, default=EVAL_ORDER_LIMIT)
    ap.add_argument("--num-drivers", type=int, default=EVAL_NUM_DRIVERS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--model", default=None, help="override LLMConfig.model")
    args = ap.parse_args()

    profile_regime = args.profile_regime or args.regimes[0]

    # Build the client FIRST so a missing key fails fast, before env construction.
    cfg = LLMConfig()
    if args.model:
        cfg.model = args.model
    client = make_llm_client(cfg)  # raises a helpful error if the key is missing

    env = make_nyc_env(
        seed=args.seed, regime=profile_regime, split=args.split,
        num_drivers=args.num_drivers, order_limit=args.order_limit,
    )
    env.reset(seed=args.seed)

    def dist(a, b):
        return env.network.shortest_path(a, b).travel_time

    phi = GlobalStats.from_observations(env._build_observations(), dist=dist)
    profile = encode_env_profile(
        env, phi, profile_regime, args.split, random.Random(args.seed), dist=dist
    )

    best = evolve_pure_phase1(
        client, profile,
        generations=args.generations, lam=args.lam,
        regimes=tuple(args.regimes), split=args.split,
        num_drivers=args.num_drivers,
        order_limit=args.order_limit, seed=args.seed,
        temperature=args.temperature,
    )

    path = freeze_pure_phase1(best, regime="+".join(args.regimes))
    print("\n=== FROZEN (pure-Phase-1) ===")
    print("skill      :", best.name)
    print("objective  :", best.meta["objective"])
    print("description:", best.meta["description"])
    print("reward CoT :", best.meta["reward_understanding"])
    print("reward     : %.4g (fixed default-MARL income_mean)"
          % best.evaluation.fitness_mean)
    print("written to :", path)


if __name__ == "__main__":
    main()
