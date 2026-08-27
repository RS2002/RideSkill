"""One-command Phase-2 combiner evolution (§5 / milestone 7.6).

Loads the frozen skill basis (seeds + any Phase-1 evolved skills), builds the
env profile and the FIXED scalarisation frame, evolves the upper
``skill_scores`` combiner over ``W_train``, and freezes the winner to
``pref_dispatch/evolved/combiners/``.

Requires the API key in the environment / git-ignored ``.env`` (never in repo):

    python -m pref_dispatch.llm.run_evolve_combiner --n-train 6

Use ``--regimes offpeak --n-train 3 --generations 0`` for a fast wiring run.

§Phase-2 single-reward arms (one preference -> one reward -> one strategy, NO
runtime preference dial):

    # LLM authors a reward from a natural-language preference, then composes ONE
    # strategy that maximises it, frozen under its own name:
    python -m pref_dispatch.llm.run_evolve_combiner --scenarios 6 \\
        --author-reward --preference-nl "prefer completion, dislike long detours" \\
        --name reward_maximizing_dispatcher

    # Or author from an explicit metric-weight preference:
    ... --author-reward --pref-weights revenue=0.8,service=0.2 ...

    # Or take the env's OWN reward_func verbatim as the fixed objective:
    ... --reward-from-env ...

These force ``objective=env_reward`` and ``ignore_pref=True`` and require
``--scenarios`` (reward injection is only wired on the scenarios path). Feed
``--preference-nl`` multiple times to sweep OBJECTIVES: each preference authors its
OWN reward and freezes its OWN independent strategy (N preferences -> N strategies),
which is an objective scan, NOT a runtime dial.
"""

from __future__ import annotations

import argparse
import random

from pref_dispatch.global_stats import GlobalStats
from pref_dispatch.llm.basis import load_basis
from pref_dispatch.llm.client import make_llm_client
from pref_dispatch.llm.combiner_eval import (
    build_norm_frame,
    make_train_prefs,
    scenario_norm_frames,
)
from pref_dispatch.llm.config import LLMConfig
from pref_dispatch.llm.encode import encode_env_profile
from pref_dispatch.llm.evolve_combiner import evolve_combiner, freeze_combiner
from pref_dispatch.llm.evolve_reward import author_reward
from pref_dispatch.llm.fitness_eval import EVAL_NUM_DRIVERS, EVAL_ORDER_LIMIT
from pref_dispatch.llm.reward_spec import describe_reward
from pref_dispatch.nyc_env import make_nyc_env
from pref_dispatch.scenario import ScenarioRanges, sample_scenario_set


def _reward_coeff_snapshot(reward_fn) -> dict:
    """Live coefficient snapshot of the reward the combiner is composed FOR."""
    keys = ("assignment_bonus", "revenue_coef", "service_time_coef",
            "detour_coef", "empty_move_penalty", "idle_penalty")
    return {k: getattr(reward_fn, k) for k in keys if hasattr(reward_fn, k)}


def _parse_pref_weights(spec: str) -> dict:
    """Parse ``revenue=0.8,service=0.2`` into a weight dict for reward authoring."""
    out = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        k, _, v = item.partition("=")
        out[k.strip()] = float(v)
    if not out:
        raise ValueError(f"empty --pref-weights: {spec!r}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Evolve the Phase-2 upper combiner.")
    ap.add_argument("--regimes", nargs="+", default=["offpeak", "shoulder", "peak"])
    ap.add_argument("--profile-regime", default=None,
                    help="regime used to build the env profile (default: first).")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-train", type=int, default=6, help="|W_train| preferences.")
    ap.add_argument("--generations", type=int, default=4)
    ap.add_argument("--lam", type=int, default=2)
    ap.add_argument("--order-limit", type=int, default=EVAL_ORDER_LIMIT)
    ap.add_argument("--num-drivers", type=int, default=EVAL_NUM_DRIVERS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--no-evolved", action="store_true",
                    help="use only the handwritten seeds as the frozen basis.")
    ap.add_argument("--model", default=None, help="override LLMConfig.model")
    # v2 domain-randomized generalization evolution.
    ap.add_argument("--scenarios", type=int, default=0,
                    help="if >0, evolve the combiner on this many RANDOM "
                         "domain-randomized scenarios (v2). 0 = v1 fixed point.")
    # Reward-conditioned arm: read the env's ACTUAL reward function, hand it to the
    # LLM (which must first EXPLAIN it), and score by the env reward itself.
    ap.add_argument("--reward-conditioned", action="store_true",
                    help="compose the combiner FOR the env's current reward_func: "
                         "inject a REWARD FUNCTION section + reward_understanding "
                         "CoT gate, and use objective=env_reward (maximise the "
                         "actual DefaultRewardFunction fleet mean).")
    # §Phase-2 single-reward arms: one preference -> one reward -> one strategy.
    ap.add_argument("--author-reward", action="store_true",
                    help="LLM AUTHORS a reward_func from a preference (--preference-nl "
                         "or --pref-weights), then composes ONE strategy that maximises "
                         "it (objective=env_reward, ignore_pref). Requires --scenarios.")
    ap.add_argument("--preference-nl", action="append", default=None,
                    help="natural-language platform preference to author a reward from. "
                         "Repeatable: each preference authors its OWN reward and freezes "
                         "its OWN independent strategy (an OBJECTIVE sweep, not a dial).")
    ap.add_argument("--pref-weights", default=None,
                    help="metric-weight preference, e.g. 'revenue=0.8,service=0.2', "
                         "authored into a reward (alternative to --preference-nl).")
    ap.add_argument("--reward-from-env", action="store_true",
                    help="take the env's OWN reward_func verbatim as the fixed "
                         "objective (no authoring); compose ONE strategy for it "
                         "(objective=env_reward, ignore_pref). Requires --scenarios.")
    ap.add_argument("--name", default=None,
                    help="override the frozen combiner name (e.g. "
                         "reward_maximizing_dispatcher).")
    args = ap.parse_args()

    profile_regime = args.profile_regime or args.regimes[0]
    use_scenarios = args.scenarios > 0
    # The single-reward arms (author / from-env) compose for ONE fixed reward with
    # NO preference dial -> objective=env_reward + ignore_pref. --reward-conditioned
    # is the legacy v2 arm (reward + runtime slider) and stays as-is.
    single_reward = args.author_reward or args.reward_from_env
    ignore_pref = single_reward
    objective = "env_reward" if (args.reward_conditioned or single_reward) else "scalarize"

    if single_reward and not use_scenarios:
        ap.error("--author-reward / --reward-from-env require --scenarios > 0 "
                 "(reward injection is wired only on the scenarios path).")

    # Build the client FIRST so a missing key fails fast.
    cfg = LLMConfig()
    if args.model:
        cfg.model = args.model
    client = make_llm_client(cfg)

    skills, cards = load_basis(include_evolved=not args.no_evolved)
    print("frozen basis:", list(skills))

    env = make_nyc_env(
        seed=args.seed, regime=profile_regime, split=args.split,
        num_drivers=args.num_drivers, order_limit=args.order_limit,
    )
    env.reset(seed=args.seed)

    def dist(a, b):
        return env.network.shortest_path(a, b).travel_time

    phi = GlobalStats.from_observations(env._build_observations(), dist=dist)
    ranges_env = ScenarioRanges(order_limit=args.order_limit) if use_scenarios else None
    profile = encode_env_profile(
        env, phi, profile_regime, args.split, random.Random(args.seed), dist=dist,
        ranges=ranges_env, prev_windows=(1, 2) if use_scenarios else (1,),
    )

    # Reward jobs: each is one fixed objective the combiner is composed FOR.
    # Legacy/scalarize arms -> a single job with no reward injection. The
    # single-reward arms build ONE job per preference (author) or the env reward
    # (from-env); each job freezes its OWN independent strategy.
    reward_jobs = []  # list of dict(spec_text, reward_function, snapshot, provenance)

    if args.reward_conditioned:
        # Legacy v2 arm: reward + runtime slider (objective=env_reward, keep pref).
        reward_jobs.append({
            "spec_text": describe_reward(env.reward_function),
            "reward_function": env.reward_function,
            "snapshot": _reward_coeff_snapshot(env.reward_function),
            "provenance": None,
        })
        print("reward-conditioned: composing FOR the env reward_func "
              f"({type(env.reward_function).__name__}); objective=env_reward")
        print("reward coefficients:", reward_jobs[-1]["snapshot"])
    elif args.reward_from_env:
        # Single-reward arm: env's own reward verbatim as the fixed objective.
        authored = author_reward(client, env.reward_function)
        reward_jobs.append({
            "spec_text": authored.spec_text,
            "reward_function": authored.fn,
            "snapshot": _reward_coeff_snapshot(env.reward_function),
            "provenance": authored.meta,
        })
        print("reward-from-env: composing ONE strategy FOR the env reward_func "
              f"({type(env.reward_function).__name__}); objective=env_reward, ignore_pref")
    elif args.author_reward:
        # Single-reward arm: LLM authors a reward per preference (objective sweep).
        preferences = list(args.preference_nl or [])
        if args.pref_weights:
            preferences.append(_parse_pref_weights(args.pref_weights))
        if not preferences:
            ap.error("--author-reward needs --preference-nl and/or --pref-weights.")
        for pref in preferences:
            authored = author_reward(client, pref)
            reward_jobs.append({
                "spec_text": authored.spec_text,
                "reward_function": authored.fn,
                "snapshot": None,
                "provenance": authored.meta,
            })
        print(f"author-reward: {len(reward_jobs)} authored reward(s) -> "
              f"{len(reward_jobs)} independent strateg(y/ies); objective=env_reward, "
              "ignore_pref")
    else:
        # Legacy scalarize (or plain env_reward without a reward job): no injection.
        reward_jobs.append({
            "spec_text": None, "reward_function": None,
            "snapshot": None, "provenance": None,
        })

    scenarios = frames = train_prefs = ranges = None
    if use_scenarios:
        scenarios = sample_scenario_set(
            args.scenarios, seed=args.seed + 1, ranges=ranges_env, split=args.split,
        )
    else:
        train_prefs = make_train_prefs(n=args.n_train, seed=args.seed + 1)
        print(f"building fixed scalarisation frame from {len(skills)} skills x "
              f"{len(args.regimes)} regimes ...")
        ranges = build_norm_frame(
            skills, train_prefs, regimes=tuple(args.regimes), split=args.split,
            num_drivers=args.num_drivers, order_limit=args.order_limit, seed=args.seed,
        )

    # One independent evolve+freeze per reward job (N preferences -> N strategies).
    for job_i, job in enumerate(reward_jobs):
        rfn = job["reward_function"]
        # Per-scenario frames must be measured under THIS job's reward so income_mean
        # lo/hi are source-matched to the objective the combiner is scored by.
        if use_scenarios:
            print(f"[job {job_i}] building per-scenario frames for {len(scenarios)} "
                  f"scenarios x {len(skills)} skills ...")
            frames = scenario_norm_frames(skills, scenarios, reward_function=rfn)

        best = evolve_combiner(
            client, profile, skills, cards, train_prefs, ranges,
            scenarios=scenarios, scenario_frames=frames,
            reward_spec=job["spec_text"], reward_function=rfn,
            ignore_pref=ignore_pref, objective=objective,
            generations=args.generations, lam=args.lam,
            regimes=tuple(args.regimes), split=args.split,
            num_drivers=args.num_drivers, order_limit=args.order_limit,
            seed=args.seed, temperature=args.temperature,
        )

        # Optional explicit name override (single reward job only, else keep the
        # LLM's per-strategy name so N jobs don't clobber one file).
        if args.name and len(reward_jobs) == 1:
            best.meta["combiner_name"] = args.name

        path = freeze_combiner(
            best, reward_snapshot=job["snapshot"],
            reward_provenance=job["provenance"], ignore_pref=ignore_pref,
        )
        print(f"\n=== FROZEN COMBINER [job {job_i}] ===")
        print("name       :", best.name)
        print("strategy   :", best.meta["strategy"])
        print("description:", best.meta["description"])
        if job["provenance"] and job["provenance"].get("reward_name"):
            print("target rwd :", job["provenance"]["reward_name"],
                  "|", job["provenance"].get("objective", ""))
        if best.meta.get("reward_understanding"):
            print("reward CoT :", best.meta["reward_understanding"])
        print("fitness    : %.4g (raw %.4g, fallback %.2f)" % (
            best.evaluation.fitness, best.evaluation.raw_fitness,
            best.evaluation.fallback_rate,
        ))
        if best.evaluation.per_regime:
            print("per regime :", {r: round(v, 3) for r, v in best.evaluation.per_regime.items()})
        print("written to :", path)


if __name__ == "__main__":
    main()
