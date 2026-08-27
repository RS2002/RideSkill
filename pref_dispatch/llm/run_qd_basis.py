"""One-command Phase-1 QD basis discovery (§4.7 / milestone 7.5).

Seeds the basis with the three handwritten extremes (revenue / service / enroute),
then runs the novelty-search loop: the LLM self-invents new objectives, each is
evolved and kept only if its behavioural signature is distinct (cosine <= tau)
from every skill already in the basis. Accepted skills are frozen to
``pref_dispatch/evolved/skills/``.

Requires the API key in the environment / git-ignored ``.env`` (never in the repo):

    python -m pref_dispatch.llm.run_qd_basis --max-skills 6

Use ``--regimes offpeak --generations 0 --max-skills 4`` for a fast wiring run.
"""

from __future__ import annotations

import argparse
import random

from pref_dispatch.global_stats import GlobalStats
from pref_dispatch.llm.client import make_llm_client
from pref_dispatch.llm.config import LLMConfig
from pref_dispatch.llm.encode import encode_env_profile
from pref_dispatch.llm.fitness_eval import EVAL_NUM_DRIVERS, EVAL_ORDER_LIMIT
from pref_dispatch.llm.qd_basis import (
    DEFAULT_DRY_ROUNDS,
    DEFAULT_MAX_ROUNDS,
    DEFAULT_MAX_SKILLS,
    DEFAULT_SCENARIOS_PER_ROUND,
    DEFAULT_SIG_SCENARIOS,
    DEFAULT_TAU,
    discover_basis,
)
from pref_dispatch.nyc_env import make_nyc_env
from pref_dispatch.scenario import ScenarioRanges, ScenarioSampler
from pref_dispatch.skills import EnRouteSkill, RevenueSkill, ServiceSkill

_SEED_SKILLS = [RevenueSkill(), ServiceSkill(), EnRouteSkill()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Discover the Phase-1 QD skill basis.")
    ap.add_argument("--regimes", nargs="+", default=["offpeak", "shoulder", "peak"])
    ap.add_argument("--profile-regime", default=None,
                    help="regime used to build the env profile (default: first).")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-skills", type=int, default=DEFAULT_MAX_SKILLS)
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU,
                    help="behavioural-signature redundancy threshold (cosine).")
    ap.add_argument("--dry-rounds", type=int, default=DEFAULT_DRY_ROUNDS,
                    help="stop after this many consecutive redundant rounds.")
    ap.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS,
                    help="hard cap on total proposals; give up & keep after it.")
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--lam", type=int, default=2)
    ap.add_argument("--order-limit", type=int, default=EVAL_ORDER_LIMIT)
    ap.add_argument("--num-drivers", type=int, default=EVAL_NUM_DRIVERS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--model", default=None, help="override LLMConfig.model")
    # v2 domain-randomized generalization evolution.
    ap.add_argument("--scenarios", type=int, default=0,
                    help="if >0, evolve on this many RANDOM domain-randomized "
                         "scenarios per round (v2 generalization). 0 = v1 fixed point.")
    ap.add_argument("--sig-scenarios", type=int, default=DEFAULT_SIG_SCENARIOS,
                    help="fixed scenario batch size for behavioural signatures (v2).")
    ap.add_argument("--rescale", choices=["reference", "fleet_orders", "none"],
                    default="reference", help="scenario fitness rescaling mode (v2).")
    args = ap.parse_args()

    profile_regime = args.profile_regime or args.regimes[0]
    use_scenarios = args.scenarios > 0

    # Build the client FIRST so a missing key fails fast, before the expensive
    # env/profile construction.
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
    # v2: hand the LLM the deployment-variability envelope + prev 1h/2h demand +
    # simulated clock so the evolved skill is written to be scale-invariant.
    ranges = ScenarioRanges(order_limit=args.order_limit) if use_scenarios else None
    sampler = (
        ScenarioSampler(ranges=ranges, rng=random.Random(args.seed), split=args.split)
        if use_scenarios else None
    )
    profile = encode_env_profile(
        env, phi, profile_regime, args.split, random.Random(args.seed), dist=dist,
        ranges=ranges, prev_windows=(1, 2) if use_scenarios else (1,),
    )

    result = discover_basis(
        client, profile, _SEED_SKILLS,
        max_skills=args.max_skills, tau=args.tau, max_dry_rounds=args.dry_rounds,
        max_rounds=args.max_rounds,
        generations=args.generations, lam=args.lam,
        regimes=tuple(args.regimes), split=args.split,
        num_drivers=args.num_drivers, order_limit=args.order_limit,
        seed=args.seed, temperature=args.temperature,
        sampler=sampler, scenarios_per_round=args.scenarios,
        n_sig_scenarios=args.sig_scenarios, rescale=args.rescale,
    )

    print("\n=== QD BASIS ===")
    print("size        :", len(result.basis), f"({result.n_evolved} evolved + "
          f"{len(result.basis) - result.n_evolved} seeds)")
    print("stopped on  :", result.stop_reason, f"(after {result.rounds_used} rounds)")
    for b in result.basis:
        tag = "seed " if b.provenance == "seed" else "LLM  "
        print(f"  [{tag}] {b.name:<20s} {b.objective}")
        if b.frozen_path:
            print(f"           frozen -> {b.frozen_path}")


if __name__ == "__main__":
    main()
