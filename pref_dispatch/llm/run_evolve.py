"""One-command Phase-1 skill evolution (§4 / milestone 7.4).

Builds the real-Manhattan env + profile, makes the LLM client, evolves ONE skill
toward a target objective (seeded by a handwritten specialist), and freezes the
winner to ``pref_dispatch/evolved/skills/``.

Requires the API key in the environment (never in the repo):

    export YIBU_API_KEY=sk-...
    python -m pref_dispatch.llm.run_evolve --objective revenue --regimes offpeak

Use ``--regimes offpeak --order-limit 300`` for a fast wiring run, or the default
(all three regimes, order_limit 2000) for a real evolution.
"""

from __future__ import annotations

import argparse
import random

from pref_dispatch.global_stats import GlobalStats
from pref_dispatch.llm.client import make_llm_client
from pref_dispatch.llm.config import LLMConfig
from pref_dispatch.llm.encode import encode_env_profile
from pref_dispatch.llm.evolve import evolve_one_skill, freeze_skill
from pref_dispatch.llm.fitness_eval import EVAL_NUM_DRIVERS, EVAL_ORDER_LIMIT
from pref_dispatch.nyc_env import make_nyc_env
from pref_dispatch.skills import EnRouteSkill, RevenueSkill, ServiceSkill

# Seed specialist + a natural-language objective hint per target.
_SEEDS = {
    "revenue": (RevenueSkill(), "maximise platform revenue (served trip-minutes)"),
    "service": (ServiceSkill(), "minimise passenger service time (pickup + ride)"),
    "enroute": (EnRouteSkill(), "protect committed onboard orders near their deadline"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Evolve one Phase-1 skill.")
    ap.add_argument("--objective", default="revenue", choices=sorted(_SEEDS))
    ap.add_argument("--regimes", nargs="+", default=["offpeak", "shoulder", "peak"])
    ap.add_argument("--profile-regime", default=None,
                    help="regime used to build the env profile (default: first).")
    ap.add_argument("--split", default="train")
    ap.add_argument("--generations", type=int, default=4)
    ap.add_argument("--lam", type=int, default=2)
    # Evaluation operating point: the FULL real hour (order-limit None) against a
    # deliberately reduced fleet, so supply scarcity exposes the skill's
    # objective trade-off. At the deployment fleet (~1000) every skill serves
    # every order and fitness cannot separate candidates. See fitness_eval.
    ap.add_argument("--order-limit", type=int, default=EVAL_ORDER_LIMIT)
    ap.add_argument("--num-drivers", type=int, default=EVAL_NUM_DRIVERS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--model", default=None, help="override LLMConfig.model")
    args = ap.parse_args()

    profile_regime = args.profile_regime or args.regimes[0]

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
    profile = encode_env_profile(
        env, phi, profile_regime, args.split, random.Random(args.seed), dist=dist
    )

    seed_skill, hint = _SEEDS[args.objective]
    best = evolve_one_skill(
        client, profile,
        objective_hint=hint,
        reference=seed_skill,
        generations=args.generations, lam=args.lam,
        regimes=tuple(args.regimes), split=args.split,
        num_drivers=args.num_drivers,
        order_limit=args.order_limit, seed=args.seed,
        temperature=args.temperature,
    )

    path = freeze_skill(best, regime="+".join(args.regimes))
    print("\n=== FROZEN ===")
    print("skill      :", best.name)
    print("objective  :", best.meta["objective"])
    print("description:", best.meta["description"])
    print("fitness    : %.4g (delta vs seed %s)" % (
        best.evaluation.fitness_mean,
        f"{best.evaluation.delta_vs_reference:+.4g}"
        if best.evaluation.delta_vs_reference is not None else "n/a",
    ))
    print("written to :", path)
    if best.evaluation.delta_vs_reference is not None:
        verdict = "PASS" if best.evaluation.delta_vs_reference >= 0 else "BELOW SEED"
        print(f"seed check : {verdict} (evolved skill vs handwritten baseline)")


if __name__ == "__main__":
    main()
