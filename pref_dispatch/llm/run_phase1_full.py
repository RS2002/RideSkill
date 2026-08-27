"""One-command Phase-1 driver for the final version (Part B, §B1).

Runs the full skill-repository build: researcher DIRECTIONS (1a, LLM authors a
KPI-grounded fitness per direction incl. a standalone FAIRNESS skill) -> QD
self-invention + behavioural dedup (1b/1c). Both steps run the same
``(mu+lambda)`` within-scenario-GRPO search over REAL full-hour demand windows
whose fleet sizes are stratified across the shared fleet bands, so a skill is
selected for working at scarcity AND at scale rather than on the batch it got.
Accepted skills are frozen to ``pref_dispatch/evolved/skills/`` (the 3 handwritten
seeds are ALWAYS merged in at load time by
:func:`pref_dispatch.llm.basis.load_basis`, so they need not be re-frozen here).

Requires the API key in the environment / git-ignored ``.env`` (never in the repo):

    # fast path-check (no artifacts written): exercises the live QD handoff
    python -m pref_dispatch.llm.run_phase1_full --num-drivers 120 --order-limit 150 \
        --scenarios 2 --sig-scenarios 2 --generations 0 --max-skills 2 \
        --one-direction --regimes offpeak --no-freeze

    # overnight full run (real full hours, 200-1500 cars, all cores)
    python -m pref_dispatch.llm.run_phase1_full --scenarios 6 --sig-scenarios 4 \
        --generations 5 --max-skills 10 --workers 14 --run-tag myrun

    # resume that run after it died: directions that finished all --generations are
    # loaded from cache/phase1_ckpt/myrun/ instead of being evolved again
    python -m pref_dispatch.llm.run_phase1_full --scenarios 6 --sig-scenarios 4 \
        --generations 5 --max-skills 10 --workers 14 --run-tag myrun --resume myrun
"""

from __future__ import annotations

import argparse
import os
import random

from pref_dispatch.global_stats import GlobalStats
from pref_dispatch.llm.batch_pairing import DEFAULT_FLEET_BANDS, BandedWindowSampler
from pref_dispatch.llm.checkpoint import LeaderCheckpoint
from pref_dispatch.llm.client import make_llm_client
from pref_dispatch.llm.config import LLMConfig
from pref_dispatch.llm.encode import encode_env_profile
from pref_dispatch.llm.evolve_skill_group import DEFAULT_FAMILY_BETA
from pref_dispatch.llm.fitness_eval import EVAL_NUM_DRIVERS, EVAL_ORDER_LIMIT
from pref_dispatch.llm.parallel import resolve_workers
from pref_dispatch.llm.qd_basis import (
    DEFAULT_DRY_ROUNDS,
    DEFAULT_MAX_ROUNDS,
    DEFAULT_MAX_SKILLS,
    DEFAULT_MIN_GAIN,
    DEFAULT_TAU,
)
from pref_dispatch.llm.run_phase1 import DEFAULT_DIRECTIONS, run_phase1
from pref_dispatch.llm.skill_audit import DEFAULT_MAX_REAUTHOR
from pref_dispatch.nyc_env import make_nyc_env
from pref_dispatch.scenario import ScenarioRanges, ScenarioSampler


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-1 skill repository (B1).")
    ap.add_argument("--regimes", nargs="+", default=["offpeak", "shoulder", "peak"])
    ap.add_argument("--profile-regime", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-skills", type=int, default=DEFAULT_MAX_SKILLS,
                    help="repository size N: filled to N, then HELD at N by "
                         "redundancy-based replacement.")
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU)
    ap.add_argument("--max-dry-rounds", type=int, default=DEFAULT_DRY_ROUNDS,
                    help="stop after R consecutive rounds that changed nothing.")
    ap.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS,
                    help="hard cap on total self-invention proposals.")
    ap.add_argument("--min-gain", type=float, default=DEFAULT_MIN_GAIN,
                    help="redundancy margin a newcomer must beat to evict.")
    ap.add_argument("--generations", type=int, default=5)
    ap.add_argument("--patience", type=int, default=0,
                    help="adaptive stop inside each skill search: stop when the "
                         "SAME variant has led this many CONSECUTIVE rounds. "
                         "0 (default) = fixed length (the v10 Phase-1 behaviour).")
    ap.add_argument("--min-gen", type=int, default=0,
                    help="minimum rounds before the adaptive stop may fire inside "
                         "each skill search. When >0 the search runs at least this "
                         "many rounds even if the leader is stable, and may only "
                         "stop early at/after round min-gen once the leader has "
                         "held for `patience` consecutive rounds. 0 = no minimum.")
    ap.add_argument("--runoff", action="store_true",
                    help="runoff final inside each skill search: every round's "
                         "leading variant (deduplicated by code) is re-rolled on "
                         "ONE fresh scene batch and that single paired GRPO "
                         "comparison picks the frozen variant. Off by default "
                         "(the v10 Phase-1 behaviour).")
    ap.add_argument("--mu", type=int, default=3,
                    help="survivors kept per generation ((mu+lambda) search).")
    ap.add_argument("--lam", type=int, default=4,
                    help="children produced per generation.")
    ap.add_argument("--crossover-rate", type=float, default=0.35,
                    help="share of children built by COMBINING two survivors.")
    ap.add_argument("--fresh-per-round", type=int, default=1,
                    help="children per generation written from scratch (no parent).")
    ap.add_argument("--band-beta", type=float, default=DEFAULT_FAMILY_BETA,
                    help="weight on a variant's WEAKEST fleet band in selection.")
    ap.add_argument("--workers", type=int, default=None,
                    help="processes the rollouts are spread over. Default = cores "
                         "minus 2; 1 = single-process.")
    ap.add_argument("--scenarios", type=int, default=6,
                    help="real full-hour windows per generation, stratified over "
                         "the fleet bands (use a multiple of 3 for even coverage).")
    ap.add_argument("--sig-scenarios", type=int, default=4,
                    help="fixed scenario batch size for behavioural signatures.")
    ap.add_argument("--rescale", choices=["reference", "fleet_orders", "none"],
                    default="reference")
    ap.add_argument("--num-drivers", type=int, default=EVAL_NUM_DRIVERS)
    ap.add_argument("--order-limit", type=int, default=EVAL_ORDER_LIMIT,
                    help="orders in the PROFILE env only; training windows are "
                         "always full real hours.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--model", default=None)
    ap.add_argument("--llm-timeout", type=float, default=None,
                    help="per-call LLM timeout in seconds (default: LLMConfig's "
                         "120). Slow models (e.g. deepseek reasoning models "
                         "through the gateway) need 300+ for a full skill "
                         "generation; the ds1 run produced 0/10 skills because "
                         "every call hit the 125s wall-clock guard.")
    ap.add_argument("--one-direction", action="store_true",
                    help="use only the first researcher direction (fast path-check).")
    ap.add_argument("--run-tag", default=None,
                    help="subdirectory name for this run's leader checkpoints "
                         "(default: a timestamp).")
    ap.add_argument("--no-ckpt", action="store_true",
                    help="do not write per-generation leader checkpoints under cache/.")
    ap.add_argument("--resume", default=None, metavar="RUN_TAG",
                    help="reuse an earlier run's leader checkpoints "
                         "(cache/phase1_ckpt/RUN_TAG/): every DIRECTION that "
                         "finished all --generations is loaded instead of "
                         "re-evolved; unfinished ones are searched again. The QD "
                         "stage always restarts (its repository state is not "
                         "checkpointed). Pass the SAME value as --run-tag to keep "
                         "extending one run's checkpoints.")
    ap.add_argument("--no-self-invention", dest="self_invention",
                    action="store_false", help="stop after 1a directed skills.")
    ap.add_argument("--max-reauthor", type=int, default=DEFAULT_MAX_REAUTHOR,
                    help="how many times a finished search may be REDONE after the "
                         "post-search audit rules that the fitness the model "
                         "authored does not ask for the intended behaviour "
                         "(searches = 1 + this). Past the cap the last champion is "
                         "frozen with audit.status='unresolved'.")
    ap.add_argument("--no-audit", dest="audit", action="store_false",
                    help="skip the post-search audit entirely (one search per "
                         "skill, nothing judged). Only Phase 1 has this check: it "
                         "is the only phase whose fitness the model writes.")
    ap.add_argument("--no-freeze", dest="freeze", action="store_false",
                    help="do not write artifacts (path-check).")
    args = ap.parse_args()

    order_limit = None if (args.order_limit is not None and args.order_limit < 0) \
        else args.order_limit
    profile_regime = args.profile_regime or args.regimes[0]

    # Build the client FIRST so a missing key fails fast.
    cfg = LLMConfig()
    if args.model:
        cfg.model = args.model
    if args.llm_timeout:
        cfg.timeout = float(args.llm_timeout)
    client = make_llm_client(cfg)

    workers = resolve_workers(args.workers)
    print(f"[phase1] rolling on {workers} process(es) "
          f"({os.cpu_count()} logical cores visible)")

    env = make_nyc_env(
        seed=args.seed, regime=profile_regime, split=args.split,
        num_drivers=args.num_drivers, order_limit=order_limit,
    )
    env.reset(seed=args.seed)

    def dist(a, b):
        return env.network.shortest_path(a, b).travel_time

    phi = GlobalStats.from_observations(env._build_observations(), dist=dist)
    # v6 operating point: no order cap anywhere in TRAINING. Capping the stream
    # kept the earliest N requests, i.e. a ~3-minute rush and then 57 minutes of
    # an empty city -- an hour that does not occur in the data and that a skill
    # can win by rules that are useless on a real one. Fleet stays log-uniform
    # inside each band so the scarce end is not a rare tail.
    ranges = ScenarioRanges(order_limit=None, fleet_dist="loguniform")
    base_sampler = ScenarioSampler(ranges=ranges, rng=random.Random(args.seed),
                                   split=args.split)
    sampler = BandedWindowSampler(base_sampler, bands=DEFAULT_FLEET_BANDS,
                                  min_high_volume=1, ranges=ranges)
    print(f"[phase1] scene bands: {DEFAULT_FLEET_BANDS} "
          f"({args.scenarios} full real hours per generation)")
    profile = encode_env_profile(
        env, phi, profile_regime, args.split, random.Random(args.seed), dist=dist,
        ranges=ranges, prev_windows=(1, 2),
    )

    directions = DEFAULT_DIRECTIONS[:1] if args.one_direction else DEFAULT_DIRECTIONS

    ckpt = None if args.no_ckpt else LeaderCheckpoint(1, run=args.run_tag)
    if ckpt is not None:
        print(f"[phase1] per-generation leader checkpoints -> {ckpt.dir}")

    res = run_phase1(
        client, profile,
        directions=directions,
        sampler=sampler,
        scenarios_per_round=args.scenarios,
        n_sig_scenarios=args.sig_scenarios,
        rescale=args.rescale,
        max_skills=args.max_skills,
        tau=args.tau,
        max_dry_rounds=args.max_dry_rounds,
        max_rounds=args.max_rounds,
        min_gain=args.min_gain,
        run_self_invention=args.self_invention,
        generations=args.generations,
        lam=args.lam,
        mu=args.mu,
        crossover_rate=args.crossover_rate,
        fresh_per_round=args.fresh_per_round,
        band_beta=args.band_beta,
        bands=DEFAULT_FLEET_BANDS,
        workers=workers,
        checkpoint_fn=ckpt,
        resume_run=args.resume,
        regimes=tuple(args.regimes),
        split=args.split,
        num_drivers=args.num_drivers,
        order_limit=order_limit,
        seed=args.seed,
        temperature=args.temperature,
        max_reauthor=args.max_reauthor,
        audit=args.audit,
        freeze=args.freeze,
        patience=args.patience,
        min_gen=args.min_gen,
        runoff=args.runoff,
    )

    print("\n=== PHASE 1 SKILL REPOSITORY ===")
    print(f"directed skills : {res.n_directed}")
    print(f"QD evolved      : {res.n_qd_evolved}")
    print(f"total in basis  : {len(res.basis)}")
    if res.qd is not None:
        print(f"QD stop reason  : {res.qd.stop_reason} "
              f"(after {res.qd.rounds_used} rounds)")
        print(f"QD replacements : {res.qd.n_replaced} evicted-and-replaced, "
              f"{res.qd.n_rejected} proposals rejected")
    for b in res.basis:
        print(f"  [{b.provenance:8s}] {b.name:<24s} {b.objective}")
        if b.frozen_path:
            print(f"             frozen -> {b.frozen_path}")


if __name__ == "__main__":
    main()
