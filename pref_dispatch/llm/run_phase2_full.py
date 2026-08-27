"""One-command Phase-2 driver for the final version (Part B, §B2e).

Trains ONE objective-reading combiner over the FROZEN Phase-1 skill repository. The
paradigm (see :func:`pref_dispatch.llm.evolve_combiner.evolve_combiner_objectives`):
every candidate is scored across a BATCH of (real one-hour demand window, sampled
objective) pairs -- each pair injects its OWN reward into the env AND hands it to
the combiner as ``w`` -- and fitness is the GAIN OVER NOT CHOOSING, standardised by
how much the round's programs disagree about that same gain: per pair,
``(mine - equal_blend) / spread of everyone's (theirs - equal_blend)``. The mean
over the batch is the raw fitness.

The subtracted baseline is :class:`~pref_dispatch.combiner.EqualBlendCombiner` --
every frozen skill weighted the same, for every driver, under every objective --
rolled on the SAME scene and seed. It still dispatches; it just never chooses. So
0.0 means "your per-driver skill choice was worth exactly as much as not choosing"
and a negative score means choosing LOST money against a flat blend. It is also
exactly what a crashing candidate falls back to, so a program that breaks
everywhere scores 0.0 by construction and no fallback penalty is needed
(``--fallback-penalty`` defaults to 0 and survives only to reproduce old runs).

Why standardise by the round's spread rather than score raw deltas: objectives span
wildly different INTERNAL scales (a 2x weight doubles the numbers, not the
difficulty), so raw gains cannot be averaged across families -- one family's units
would decide the run. Dividing by the round's own disagreement is scale-free while
keeping the SIGN absolute, which the older centred ``(reward - mean)/std`` did not:
there, a round where everyone was useless still paid its median candidate 0.00.
Selecting for gain-over-not-choosing across a *distribution* of objectives is what
breeds the "reads ``w``, generalises to unseen objectives with zero retraining"
property. The winner is frozen to ``pref_dispatch/evolved/combiners/``.

v6 search, in four points:

* **``(mu+lambda)``-ES, not a hill-climb.** ``--mu`` survivors (plus one reserved
  elite slot per objective family, so a specialist on a hard family is not culled)
  produce ``--lam`` offspring per round; an offspring is either an LLM MUTATION of
  one survivor or -- at ``--crossover-rate`` -- an LLM CROSSOVER of two, which is
  how a specialist's mechanism reaches a strong all-rounder instead of dying with it.
* **Scenes rotate every round**: ``--objectives`` fresh real windows (uniform over
  the split's 84 train windows, >=1 from the busy 17-19h hours, always the FULL
  hour -- no order cap) paired with freshly sampled objectives.
* **Parents are re-rolled with the offspring** on that grid, so each round is an
  exact paired comparison rather than a rank against whatever scenes a candidate
  happened to be admitted on.
* **The 8 frozen single skills are out of the selection group** and run instead as
  a separate fixed-batch yardstick (``--yardstick-scenes``) -- the paper's "beats N
  of 8 single skills" number, and the only cross-round comparable point.

The skill basis is loaded via :func:`pref_dispatch.llm.basis.load_basis` (the 3
handwritten seeds + everything Phase 1 froze under ``evolved/skills/``), so run this
only AFTER Phase 1 has frozen its skills.

Requires the API key in the environment / git-ignored ``.env`` (never in the repo,
per MEMORY ``never-write-api-key-to-repo``). The objective sampler folds the LLM-only
``nl`` family in automatically once a client is present.

    # fast path-check (no artifacts written)
    python -m pref_dispatch.llm.run_phase2_full --num-drivers 120 \
        --objectives 3 --generations 0 --mu 1 --lam 1 --yardstick-scenes 0 \
        --regimes offpeak --no-freeze

    # overnight full run
    python -m pref_dispatch.llm.run_phase2_full --objectives 18 --generations 8
"""

from __future__ import annotations

import argparse
import os
import random

from pref_dispatch.global_stats import GlobalStats
from pref_dispatch.llm.basis import load_basis
# Re-exported: the pairing rule moved to its own module on 2026-08-10 so Phases 1
# and 3 can use it too. Importers of ``run_phase2_full.pair_by_fleet_band`` keep
# working.
from pref_dispatch.llm.batch_pairing import (  # noqa: F401
    DEFAULT_FLEET_BANDS,
    pair_by_fleet_band,
)
from pref_dispatch.llm.client import make_llm_client
from pref_dispatch.llm.config import LLMConfig
from pref_dispatch.llm.encode import encode_env_profile
from pref_dispatch.llm.evolve_combiner import (
    evolve_combiner_objectives,
    freeze_combiner,
    selection_score,
    skill_yardstick,
)
from pref_dispatch.llm.fitness_eval import EVAL_NUM_DRIVERS, EVAL_ORDER_LIMIT
from pref_dispatch.llm.resume import describe_run, load_leader_seed
from pref_dispatch.llm.group_fitness import (
    DEFAULT_FALLBACK_PENALTY,
    DEFAULT_FAMILY_BETA,
)
from pref_dispatch.llm.checkpoint import LeaderCheckpoint
from pref_dispatch.llm.objective_sampler import ObjectiveSampler
from pref_dispatch.llm.parallel import resolve_workers
from pref_dispatch.llm.policy_audit import (
    DEFAULT_CELLS as DEFAULT_AUDIT_CELLS,
    probe_phase2,
    run_policy_audit,
)
from pref_dispatch.nyc_env import make_nyc_env
from pref_dispatch.scenario import ScenarioRanges, ScenarioSampler


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-2 objective-reading combiner (B2).")
    ap.add_argument("--regimes", nargs="+", default=["offpeak", "shoulder", "peak"])
    ap.add_argument("--profile-regime", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--objectives", type=int, default=18,
                    help="(scene, objective) pairs drawn PER ROUND.")
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--patience", type=int, default=3,
                    help="adaptive stop: end the search when the SAME program has "
                         "led this many CONSECUTIVE rounds (converged). "
                         "--generations becomes the hard cap. 0 = fixed length.")
    ap.add_argument("--min-gen", type=int, default=0,
                    help="minimum rounds before the adaptive stop may fire. "
                         "When >0, the search runs at least this many rounds even "
                         "if the leader is stable; it can only stop early at/after "
                         "round min_gen once the leader has held for `patience` "
                         "consecutive rounds. 0 = no minimum (old behaviour).")
    ap.add_argument("--no-runoff", dest="runoff", action="store_false",
                    help="skip the runoff final. By default every round's leader "
                         "(deduplicated by code) is re-rolled on ONE fresh batch "
                         "at the end and that single paired GRPO comparison picks "
                         "the champion -- removing the last-round sampling "
                         "lottery.")
    ap.set_defaults(runoff=True)
    ap.add_argument("--mu", type=int, default=4,
                    help="survivors kept per round (plus one reserved elite slot "
                         "per objective family).")
    ap.add_argument("--lam", type=int, default=4,
                    help="offspring proposed per round.")
    ap.add_argument("--crossover-rate", type=float, default=0.35,
                    help="share of offspring built by crossing TWO survivors "
                         "instead of mutating one.")
    ap.add_argument("--skill-refs-in-group", action="store_true",
                    help="also rank against the 8 frozen single skills inside the "
                         "selection group (v5 behaviour; costs 8 extra full-hour "
                         "rollouts per pair). Off by default -- they run as the "
                         "separate fixed-batch yardstick instead.")
    ap.add_argument("--yardstick-scenes", type=int, default=12,
                    help="fixed (scene, objective) pairs the champion is ranked "
                         "against every frozen skill on at the end; 0 = skip.")
    ap.add_argument("--audit-cells", type=int, default=DEFAULT_AUDIT_CELLS,
                    help="post-search soft check: (scene, objective) cells rolled "
                         "TWICE at the end -- once with the objective w withheld "
                         "from the combiner, once with it handed over -- and read "
                         "back by the model. Costs 2 full-hour rollouts per cell. "
                         "ADVISORY ONLY: it never regrades, retries or re-freezes "
                         "anything. 0 = skip.")
    ap.add_argument("--fallback-penalty", type=float, default=0.0,
                    help="LEGACY, default off. A crash now runs the EQUAL BLEND, "
                         "which is the fitness baseline itself, so a combiner that "
                         "breaks everywhere already scores 0 with no penalty term. "
                         "Set >0 only to reproduce the pre-delta-fitness runs, where "
                         "a crash silently inherited skill_names[0] -- a working "
                         "single-skill policy -- and had to be charged for it.")
    ap.add_argument("--family-beta", type=float, default=DEFAULT_FAMILY_BETA,
                    help="family-aware selection weight: selection = mean advantage + "
                         "family_beta x weakest-family advantage (see selection_score).")
    ap.add_argument("--structural-fraction", type=float, default=0.5,
                    help="share of the objective batch reserved for the term-different "
                         "structural families (completion/pooling), split "
                         "round-robin so neither can come out 0.")
    ap.add_argument("--num-drivers", type=int, default=EVAL_NUM_DRIVERS)
    ap.add_argument("--order-limit", type=int, default=EVAL_ORDER_LIMIT,
                    help="orders per rollout; omit / -1 = full real hour.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None,
                    help="processes the rollouts are spread over. Default = cores "
                         "minus 2; 1 = single-process (the old behaviour). Ranks "
                         "are unaffected -- every rollout is seeded by its own "
                         "scenario.")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--llm-timeout", type=float, default=None,
                    help="per-call LLM timeout in seconds (default: LLMConfig's "
                         "120). Slow models (e.g. deepseek through the gateway) "
                         "need 300+ for a full generation.")
    ap.add_argument("--run-tag", default=None,
                    help="subdirectory name for this run's leader checkpoints "
                         "(default: a timestamp).")
    ap.add_argument("--no-ckpt", action="store_true",
                    help="do not write per-round leader checkpoints under cache/.")
    ap.add_argument("--resume", default=None, metavar="RUN_TAG",
                    help="WARM-START from an earlier run's leader checkpoint "
                         "(cache/phase2_ckpt/RUN_TAG/leader.json): that champion is "
                         "injected into generation 0 and must re-earn its place. "
                         "Unlike Phase 1 this does NOT skip rounds -- all "
                         "--generations are re-run, so it recovers the program, not "
                         "the elapsed search. Pass the SAME value as --run-tag to "
                         "keep extending one run's checkpoints.")
    ap.add_argument("--no-freeze", dest="freeze", action="store_false",
                    help="do not write the frozen combiner (path-check).")
    ap.add_argument("--probe-event-evolve", action="store_true", default=False,
                    help="enable probe-event evolution: instruct the LLM to "
                         "design diverse synthetic event dicts (pickup_wait, "
                         "detour, seating, etc.) rather than using a fixed "
                         "template. Produces combiners with richer objective "
                         "sensitivity without changing the skill_scores signature.")
    args = ap.parse_args()

    order_limit = None if (args.order_limit is not None and args.order_limit < 0) \
        else args.order_limit
    profile_regime = args.profile_regime or args.regimes[0]
    workers = resolve_workers(args.workers)
    print(f"[phase2] rolling on {workers} process(es) "
          f"({os.cpu_count()} logical cores visible)")

    # Training envelope (v6): log-uniform fleet so the small-scale regime is well
    # covered, and NO order cap -- every training scene is a full real demand hour
    # (sample_real_windows forces order_limit=None). --order-limit only shortens
    # the one throwaway env used to encode the profile below.
    ranges = ScenarioRanges(order_limit=None, fleet_dist="loguniform")

    # Build the client FIRST so a missing key fails fast.
    cfg = LLMConfig()
    if args.model:
        cfg.model = args.model
    if args.llm_timeout:
        cfg.timeout = float(args.llm_timeout)
    client = make_llm_client(cfg)

    # Frozen Phase-1 skill repository (3 seeds + evolved skills on disk).
    skills, cards = load_basis(include_evolved=True)
    print(f"[phase2] loaded {len(skills)} frozen skills: {list(skills)}")

    env = make_nyc_env(
        seed=args.seed, regime=profile_regime, split=args.split,
        num_drivers=args.num_drivers, order_limit=order_limit,
    )
    env.reset(seed=args.seed)

    def dist(a, b):
        return env.network.shortest_path(a, b).travel_time

    phi = GlobalStats.from_observations(env._build_observations(), dist=dist)
    scen_sampler = ScenarioSampler(ranges=ranges, rng=random.Random(args.seed),
                                   split=args.split)
    profile = encode_env_profile(
        env, phi, profile_regime, args.split, random.Random(args.seed), dist=dist,
        ranges=ranges, prev_windows=(1, 2),
    )

    # Every ROUND draws its own batch: k real one-hour demand windows (uniform over
    # the split's 84 train windows, >=1 from the busy 17-19h hours) paired with k
    # freshly sampled objectives. Rotating the scenes is what stops a candidate
    # from winning by fitting one batch of windows; every program alive in a round
    # -- surviving parents included -- is re-rolled on exactly this grid, so the
    # round's group-relative advantages are a paired comparison.
    _BANDS = DEFAULT_FLEET_BANDS
    obj_sampler = ObjectiveSampler(
        client=client, rng=random.Random(args.seed + 1),
        temperature=args.temperature,
        structural_fraction=args.structural_fraction,
    )

    def _report_batch_coverage(round_idx, scs, objs, bands):
        """Software batch-coverage audit (objective term axes + scene span), log-only.

        The SOFTWARE counterpart of the LLM batch-diversity self-check. It cannot
        judge semantics ("two objectives are the same niche reworded"), but it DOES
        measure, from ``w`` on term-isolating probes, which metric axes the batch's
        objectives actually price, and which fleet bands / regimes the scenes span.
        Nothing is rejected and nothing is sampled here -- the audit is advisory, so
        a narrow batch is reported and a check can never hang the round. A
        ``w``-blind objective scores every axis 0 and is counted separately.
        """
        try:
            from pref_dispatch.llm.batch_check import batch_coverage_summary
            cov = batch_coverage_summary(scs, objs, bands)
        except Exception as e:  # noqa: BLE001 -- coverage is advisory; never fatal
            print(f"[phase2] round {round_idx} batch-coverage check failed: {e}")
            return None
        oc = cov["objective"]
        sc = cov["scene"]
        print(f"[phase2] round {round_idx} batch coverage: "
              f"objectives price axes {oc['covered_axes']} "
              f"(of {len(oc['covered_axes']) + len(oc['missing_axes'])}; "
              f"blind={oc['n_blind']}, families={oc['family_mix']}); "
              f"scenes bands={sc['bands']} regimes={sc['regimes']} "
              f"windows={sc['n_distinct_windows']}")
        if oc["missing_axes"]:
            print(f"[phase2] round {round_idx} UNPRICED objective axes: "
                  f"{oc['missing_axes']} (an objective pricing none of these is "
                  f"invisible to a combiner that only reads them)")
        return cov

    def batch_fn(round_idx: int, _memo={}):
        if round_idx in _memo:          # main draws round 0 to size the run;
            return _memo[round_idx]     # the evolution loop asks for it again.
        scs = scen_sampler.sample_real_windows(
            args.objectives, base_seed=args.seed + 1000 * round_idx,
            min_high_volume=1, ranges=ranges,
        )
        objs = obj_sampler.sample_batch(args.objectives)
        # Pair each family across DIFFERENT fleet bands (scarcity + scale). Without
        # this the family draw is independent of the scene draw, so a family can be
        # trained entirely at one scale in a round and learn a scale-specific rule.
        objs = pair_by_fleet_band(scs, objs, _BANDS)
        _memo[round_idx] = (scs, objs)
        print(f"[phase2] round {round_idx} grid "
              f"(fleet regime window seed family):")
        for i, (sc, ob) in enumerate(zip(scs, objs)):
            print(f"  [{i}] fleet={sc.num_drivers:.0f} regime={sc.regime} "
                  f"window={sc.window} seed={sc.seed} family={ob.family}")
        _report_batch_coverage(round_idx, scs, objs, _BANDS)
        return scs, objs

    scenarios, objectives = batch_fn(0)

    ckpt = None if args.no_ckpt else LeaderCheckpoint(2, run=args.run_tag)
    if ckpt is not None:
        print(f"[phase2] per-round leader checkpoints -> {ckpt.dir}")

    # --resume: warm-start gen 0 from an earlier run's checkpointed champion. This
    # recovers the PROGRAM, not the elapsed rounds -- see resume.load_leader_seed.
    seed_meta = None
    if args.resume:
        for line in describe_run(args.resume, phase=2):
            print(line)
        seed_meta = load_leader_seed(2, args.resume)

    best = evolve_combiner_objectives(
        client, profile, skills, cards, scenarios, objectives,
        batch_fn=batch_fn,
        generations=args.generations, mu=args.mu, lam=args.lam,
        crossover_rate=args.crossover_rate,
        rng=random.Random(args.seed + 7),
        skill_refs_in_group=args.skill_refs_in_group,
        temperature=args.temperature,
        fallback_penalty=args.fallback_penalty,
        family_beta=args.family_beta,
        workers=workers,
        checkpoint_fn=ckpt,
        seed_code=(seed_meta or {}).get("code"),
        seed_meta=seed_meta,
        patience=args.patience,
        runoff=args.runoff,
        min_gen=args.min_gen,
        probe_event_evolve=args.probe_event_evolve,
    )

    print("\n=== PHASE 2 COMBINER ===")
    print(f"name     : {best.name}")
    print(f"strategy : {best.meta['strategy']}")
    print(f"operator : {best.meta.get('operator', 'seed/propose')}")
    print(f"selection: {selection_score(best.evaluation, beta=args.family_beta):.3f} "
          f"= fitness {best.evaluation.fitness:.4g} + {args.family_beta} x weakest "
          f"family (family-aware key; see selection_score)")
    print(f"fitness  : {best.evaluation.fitness:.4g} "
          f"(GROUP-RELATIVE advantage {best.evaluation.raw_fitness:+.2f} "
          f"= (reward - the FINAL round's mean) / its spread, per objective, "
          f"averaged; fallback {best.evaluation.fallback_rate:.2f}, "
          f"objective blindness {best.evaluation.objective_blindness:.2f} [report: "
          f"0 = the fleet's skill mix moves between objectives, 1 = never moves])")
    fam = " ".join(f"{k} {v:+.2f}" for k, v in sorted(best.evaluation.per_family.items()))
    print(f"per-family advantage: {fam}")
    if best.evaluation.per_family:
        print(f"weakest family : {min(best.evaluation.per_family, key=best.evaluation.per_family.get)} "
              f"({min(best.evaluation.per_family.values()):.2f})")

    # The single skills left the selection group in v6; they come back HERE, on one
    # FIXED batch, as the only externally-interpretable scale the paper has.
    if args.yardstick_scenes > 0:
        y_scen = ScenarioSampler(
            ranges=ranges, rng=random.Random(args.seed + 99), split=args.split,
        ).sample_real_windows(args.yardstick_scenes, base_seed=args.seed + 99,
                              min_high_volume=2, ranges=ranges)
        y_obj = ObjectiveSampler(
            client=client, rng=random.Random(args.seed + 99),
            temperature=args.temperature,
            structural_fraction=args.structural_fraction,
        ).sample_batch(args.yardstick_scenes)
        y_obj = pair_by_fleet_band(y_scen, y_obj, _BANDS)
        card = skill_yardstick(best, skills, y_scen, y_obj,
                               fallback_penalty=args.fallback_penalty,
                               workers=workers)
        print(f"yardstick: beats {len(card['beaten'])}/{card['n_skills']} frozen "
              f"skills on {args.yardstick_scenes} fixed pairs "
              f"(mean advantage {card['rank']:+.2f})")

    # The post-search soft check. The SAME hour, seed and env reward are rolled
    # twice; the only difference is whether the combiner was handed w. Whatever
    # separates the two KPI columns is what reading the objective bought, and the
    # model is asked to read that table itself. The objectives are the model's own
    # (the NL family), on TRAIN windows -- grading this on the evaluation grid's
    # hand-written objectives would be running the diagnostic on the test set.
    # Nothing here feeds back into anything: see pref_dispatch/llm/policy_audit.py.
    if args.audit_cells > 0 and client is not None:
        a_scen = ScenarioSampler(
            ranges=ranges, rng=random.Random(args.seed + 555), split=args.split,
        ).sample_real_windows(args.audit_cells, base_seed=args.seed + 555,
                              min_high_volume=1, ranges=ranges)
        a_obj = ObjectiveSampler(
            client=client, rng=random.Random(args.seed + 555),
            temperature=args.temperature,
            family_weights={"nl": 1.0}, llm_briefs=True,
        ).sample_batch(args.audit_cells, stratify=False)
        cells = probe_phase2(best.make_combiner(), skills, a_scen, a_obj)
        best.meta["policy_audit"] = run_policy_audit(
            client, phase=2,
            objective=str(best.meta.get("strategy", "")),
            description=str(best.meta.get("description", "")),
            code=str(best.meta.get("code", "")),
            cells=cells,
        )

    if args.freeze:
        path = freeze_combiner(best)
        print(f"frozen   -> {path}")
    else:
        print("(not frozen: --no-freeze)")


if __name__ == "__main__":
    main()
