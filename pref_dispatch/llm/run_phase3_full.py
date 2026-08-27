"""One-command Phase-3 driver for the final version (Part B, §B3).

Trains ONE reposition scorer over the FROZEN Phase-1 skills + Phase-2 combiner,
with the SAME search Phase 2 uses -- see
:func:`pref_dispatch.llm.evolve_reposition.evolve_repositioner_group`. The one
extra axis is the fairness strength, so a cell here is a
``(real one-hour demand window, sampled objective, fairness strength)`` triple:
the cell injects its own reward into the env, hands it to the scorer as ``w``, and
runs the wage budget at its own strength. ONE frozen scorer therefore
best-responds ACROSS the objective and fairness axes without retraining, matching
how the budget mechanism already parameterises fairness.

Fitness is GROUP-RELATIVE (GRPO) inside a cell: ``(reward - mean)/std`` over the
programs alive that round PLUS two fixed anchors -- the built-in demand-gravity
heuristic and repositioning switched OFF. The anchors are what make the number
absolute: a round where every program is worse than not moving idle cars at all
would otherwise still hand out advantages near 0, and beating "don't move" is the
whole point of Phase 3.

Same v6/v7 search as Phase 2, in four points:

* **``(mu+lambda)``-ES, not a hill-climb.** ``--mu`` survivors (plus one reserved
  elite slot per objective family AND per strength band, so a specialist on a hard
  family or on strong fairness is not culled) produce ``--lam`` offspring per
  round; an offspring is an LLM MUTATION of one survivor or -- at
  ``--crossover-rate`` -- an LLM CROSSOVER of two.
* **Cells rotate every round**: ``--cells`` fresh real windows (uniform over the
  split's train windows, >=1 from the busy 17-19h hours, always the FULL hour --
  no order cap) paired with freshly sampled objectives and strengths.
* **Parents are re-rolled with the offspring** on that grid, so each round is an
  exact paired comparison.
* **The anchors are out of the selection group's identity** and come back at the
  end on one FIXED batch (``--yardstick-scenes``) -- the paper's "beats the
  heuristic on N of M cells / beats reposition-OFF on K of M" number, and the only
  cross-round comparable point.

Both pairings are applied to the batch: :func:`pair_by_fleet_band` spreads each
objective family over fleet bands, and :func:`pair_by_strength_band` gives every
family both a budget-OFF cell and a budget-ON one while keeping the strength band
uncorrelated with the fleet band.

Run this only AFTER Phase 1 (skills) and Phase 2 (combiner) have frozen their
artifacts -- the frozen dispatch stack is held fixed while the repositioner learns.

Requires the API key in the environment / git-ignored ``.env`` (never in the repo,
per MEMORY ``never-write-api-key-to-repo``).

    # fast path-check (no artifacts written)
    python -m pref_dispatch.llm.run_phase3_full --num-drivers 120 \\
        --cells 3 --generations 0 --mu 1 --lam 1 --yardstick-scenes 0 \\
        --regimes offpeak --no-freeze

    # overnight full run
    python -m pref_dispatch.llm.run_phase3_full --cells 18 --generations 8
"""

from __future__ import annotations

import argparse
import os
import random

from pref_dispatch.global_stats import GlobalStats
from pref_dispatch.llm.basis import load_basis, load_frozen_combiner
from pref_dispatch.llm.batch_pairing import (
    DEFAULT_FLEET_BANDS,
    pair_by_fleet_band,
    pair_by_strength_band,
)
from pref_dispatch.llm.checkpoint import LeaderCheckpoint
from pref_dispatch.llm.client import make_llm_client
from pref_dispatch.llm.config import LLMConfig
from pref_dispatch.llm.encode import encode_env_profile
from pref_dispatch.llm.evolve_reposition import (
    evolve_repositioner_group,
    freeze_repositioner,
    reposition_yardstick,
    selection_score,
)
from pref_dispatch.llm.fitness_eval import EVAL_NUM_DRIVERS, EVAL_ORDER_LIMIT
from pref_dispatch.llm.group_fitness import (
    DEFAULT_FALLBACK_PENALTY,
    DEFAULT_FAMILY_BETA,
)
from pref_dispatch.llm.objective_sampler import ObjectiveSampler
from pref_dispatch.llm.parallel import resolve_workers
from pref_dispatch.llm.policy_audit import (
    DEFAULT_CELLS as DEFAULT_AUDIT_CELLS,
    probe_phase3,
    run_policy_audit,
)
from pref_dispatch.llm.reposition_adapter import GuardedScorer
from pref_dispatch.llm.reposition_eval import strength_label
from pref_dispatch.llm.resume import describe_run, load_leader_seed
from pref_dispatch.nyc_env import make_nyc_env
from pref_dispatch.scenario import ScenarioRanges, ScenarioSampler


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase-3 objective+fairness reposition scorer (B3).")
    ap.add_argument("--regimes", nargs="+", default=["offpeak", "shoulder", "peak"])
    ap.add_argument("--profile-regime", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--cells", type=int, default=18,
                    help="(scene, objective, fairness strength) cells drawn PER ROUND.")
    ap.add_argument("--triples", type=int, default=None,
                    help="deprecated alias for --cells (the old runner's name).")
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--patience", type=int, default=3,
                    help="adaptive stop: end the search when the SAME program has "
                         "led this many CONSECUTIVE rounds (converged). "
                         "--generations becomes the hard cap. 0 = fixed length.")
    ap.add_argument("--min-gen", type=int, default=0,
                    help="minimum rounds before the adaptive stop may fire. "
                         "When >0, the search runs at least this many rounds even "
                         "if the leader is stable; it can only stop early at/after "
                         "round min-gen once the leader has held for `patience` "
                         "consecutive rounds. 0 = no minimum (old behaviour).")
    ap.add_argument("--no-runoff", dest="runoff", action="store_false",
                    help="skip the runoff final. By default every round's leader "
                         "(deduplicated by code) is re-rolled on ONE fresh batch "
                         "at the end and that single paired GRPO comparison picks "
                         "the champion.")
    ap.set_defaults(runoff=True)
    ap.add_argument("--mu", type=int, default=4,
                    help="survivors kept per round (plus one reserved elite slot "
                         "per objective family and per strength band).")
    ap.add_argument("--lam", type=int, default=4,
                    help="offspring proposed per round.")
    ap.add_argument("--crossover-rate", type=float, default=0.35,
                    help="share of offspring built by crossing TWO survivors "
                         "instead of mutating one.")
    ap.add_argument("--fresh-per-round", type=int, default=1,
                    help="parentless injections among the --lam offspring, so a "
                         "genuinely new mechanism can still enter after round 0.")
    ap.add_argument("--yardstick-scenes", type=int, default=12,
                    help="fixed cells the champion is scored against the two "
                         "anchors on at the end; 0 = skip.")
    ap.add_argument("--audit-cells", type=int, default=DEFAULT_AUDIT_CELLS,
                    help="post-search soft check: cells rolled THREE times at the "
                         "end -- objective w withheld, w handed over, and w handed "
                         "over with the fairness strength forced to 0 -- and read "
                         "back by the model. Costs 3 full-hour rollouts per cell. "
                         "ADVISORY ONLY: it never regrades, retries or re-freezes "
                         "anything. 0 = skip.")
    ap.add_argument("--fallback-penalty", type=float, default=0.0,
                    help="LEGACY, default off. A crash now parks the car, so a "
                         "scorer that breaks everywhere already scores 0 (= the "
                         "do-nothing baseline) with no penalty term. Set >0 only to "
                         "reproduce the pre-delta-fitness runs, where a crash "
                         "silently borrowed the demand-gravity heuristic's decisions "
                         "and had to be charged for them.")
    ap.add_argument("--family-beta", type=float, default=DEFAULT_FAMILY_BETA,
                    help="selection = mean advantage + family_beta x weakest-family "
                         "advantage + family_beta x weakest-strength-band advantage "
                         "(see selection_score).")
    ap.add_argument("--structural-fraction", type=float, default=0.5,
                    help="share of the objective batch reserved for the "
                         "term-different structural families "
                         "(completion/pooling).")
    ap.add_argument("--combiner-name", default=None,
                    help="frozen combiner to hold fixed (default: the sole one).")
    ap.add_argument("--num-drivers", type=int, default=EVAL_NUM_DRIVERS)
    ap.add_argument("--order-limit", type=int, default=EVAL_ORDER_LIMIT,
                    help="orders in the throwaway profiling env only; training "
                         "scenes are always full real hours.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None,
                    help="processes the rollouts are spread over. Default = cores "
                         "minus 2; 1 = single-process. Ranks are unaffected -- every "
                         "rollout is seeded by its own scenario.")
    ap.add_argument("--temperature", type=float, default=0.9)
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
                         "(cache/phase3_ckpt/RUN_TAG/leader.json): that champion is "
                         "injected into generation 0 and must re-earn its place. "
                         "Like Phase 2 (and unlike Phase 1) this does NOT skip "
                         "rounds -- all --generations are re-run, so it recovers "
                         "the program, not the elapsed search. Pass the SAME value "
                         "as --run-tag to keep extending one run's checkpoints.")
    ap.add_argument("--no-freeze", dest="freeze", action="store_false",
                    help="do not write the frozen repositioner (path-check).")
    args = ap.parse_args()

    n_cells = args.triples if args.triples is not None else args.cells
    order_limit = None if (args.order_limit is not None and args.order_limit < 0) \
        else args.order_limit
    profile_regime = args.profile_regime or args.regimes[0]
    workers = resolve_workers(args.workers)
    print(f"[phase3] rolling on {workers} process(es) "
          f"({os.cpu_count()} logical cores visible)")

    # Same training envelope as Phase 2: log-uniform fleet so the scarce regime is
    # well covered, and NO order cap -- every training scene is a full real demand
    # hour (sample_real_windows forces order_limit=None). --order-limit only
    # shortens the one throwaway env used to encode the profile below.
    ranges = ScenarioRanges(order_limit=None, fleet_dist="loguniform")

    # Build the client FIRST so a missing key fails fast.
    cfg = LLMConfig()
    if args.model:
        cfg.model = args.model
    if args.llm_timeout:
        cfg.timeout = float(args.llm_timeout)
    client = make_llm_client(cfg)

    # Frozen dispatch stack, held fixed while the repositioner learns. The
    # combiner's meta carries its source, which the parallel path needs (a
    # sandbox-compiled function cannot be pickled, so each worker rebuilds it).
    skills, _cards = load_basis(include_evolved=True)
    combiner, cmeta = load_frozen_combiner(
        name=args.combiner_name, skill_names=tuple(skills)
    )
    combiner_code = cmeta.get("code")
    print(f"[phase3] loaded {len(skills)} frozen skills + combiner "
          f"{cmeta.get('combiner_name', cmeta.get('skill_name', '?'))}")
    if workers > 1 and not combiner_code:
        print("[phase3] combiner meta has no source; the parallel path will fall "
              "back to in-process rollouts")

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

    # Every ROUND draws its own cells: k real one-hour windows paired with k fresh
    # objectives and k fairness strengths. Rotating them is what stops a candidate
    # from winning by fitting one batch; every program alive in a round -- surviving
    # parents included -- is re-rolled on exactly this grid, so the round's
    # group-relative advantages are a paired comparison.
    _BANDS = DEFAULT_FLEET_BANDS
    obj_sampler = ObjectiveSampler(
        client=client, rng=random.Random(args.seed + 1),
        temperature=args.temperature,
        structural_fraction=args.structural_fraction,
    )
    strength_rng = random.Random(args.seed + 2)

    def _report_batch_coverage(round_idx, scs, objs, bands):
        """Software batch-coverage audit (objective term axes + scene span), log-only.

        This is the SOFTWARE counterpart of the LLM batch-diversity self-check: it
        cannot judge semantics ("two objectives are the same niche reworded"), but it
        DOES measure, from ``w`` on term-isolating probes, which metric axes the
        batch's objectives actually price, and which fleet bands / regimes the scenes
        span. A batch that comes back narrow is reported here and available to the
        (advisory) LLM audit; nothing is rejected, so a check can never hang the
        round. A ``w``-blind objective scores every axis 0 and is counted separately.
        """
        try:
            from pref_dispatch.llm.batch_check import batch_coverage_summary
            cov = batch_coverage_summary(scs, objs, bands)
        except Exception as e:  # noqa: BLE001 -- coverage is advisory; never fatal
            print(f"[phase3] round {round_idx} batch-coverage check failed: {e}")
            return None
        oc = cov["objective"]
        sc = cov["scene"]
        print(f"[phase3] round {round_idx} batch coverage: "
              f"objectives price axes {oc['covered_axes']} "
              f"(of {len(oc['covered_axes'])}/{len(oc['missing_axes']) + len(oc['covered_axes'])}; "
              f"blind={oc['n_blind']}, families={oc['family_mix']}); "
              f"scenes bands={sc['bands']} regimes={sc['regimes']} "
              f"windows={sc['n_distinct_windows']}")
        if oc["missing_axes"]:
            print(f"[phase3] round {round_idx} UNPRICED objective axes: "
                  f"{oc['missing_axes']} (an objective pricing none of these is "
                  f"invisible to a combiner that only reads them)")
        return cov

    def batch_fn(round_idx: int, _memo={}):
        if round_idx in _memo:          # main draws round 0 to size the run;
            return _memo[round_idx]     # the evolution loop asks for it again.
        scs = scen_sampler.sample_real_windows(
            n_cells, base_seed=args.seed + 1000 * round_idx,
            min_high_volume=1, ranges=ranges,
        )
        objs = obj_sampler.sample_batch(n_cells)
        # Spread each family over fleet bands, then over strength bands: without
        # the first a family learns one scale's rule as if universal; without the
        # second a family can be graded only with the budget off, and the round
        # cannot tell "works everywhere" from "works while fairness is at zero".
        objs = pair_by_fleet_band(scs, objs, _BANDS)
        sts = pair_by_strength_band(scs, objs, strength_rng, _BANDS)
        _memo[round_idx] = (scs, objs, sts)
        print(f"[phase3] round {round_idx} grid "
              f"(fleet regime window seed family strength):")
        for i, (sc, ob, st) in enumerate(zip(scs, objs, sts)):
            print(f"  [{i}] fleet={sc.num_drivers:.0f} regime={sc.regime} "
                  f"window={sc.window} seed={sc.seed} family={ob.family} "
                  f"strength={st:.2f} ({strength_label(st)})")
        _report_batch_coverage(round_idx, scs, objs, _BANDS)
        return scs, objs, sts

    scenarios, objectives, strengths = batch_fn(0)

    ckpt = None if args.no_ckpt else LeaderCheckpoint(3, run=args.run_tag)
    if ckpt is not None:
        print(f"[phase3] per-round leader checkpoints -> {ckpt.dir}")

    # --resume: warm-start gen 0 from an earlier run's checkpointed champion. This
    # recovers the PROGRAM, not the elapsed rounds -- see resume.load_leader_seed.
    seed_meta = None
    if args.resume:
        for line in describe_run(args.resume, phase=3):
            print(line)
        seed_meta = load_leader_seed(3, args.resume)

    best = evolve_repositioner_group(
        client, profile, skills, combiner, scenarios, objectives, strengths,
        combiner_code=combiner_code,
        batch_fn=batch_fn,
        generations=args.generations, mu=args.mu, lam=args.lam,
        crossover_rate=args.crossover_rate,
        fresh_per_round=args.fresh_per_round,
        rng=random.Random(args.seed + 7),
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
    )

    ev = best.evaluation
    print("\n=== PHASE 3 REPOSITIONER ===")
    print(f"name     : {best.name}")
    print(f"objective: {best.meta.get('objective', '?')}")
    print(f"operator : {best.meta.get('operator', 'seed/propose')}")
    print(f"selection: {selection_score(ev, beta=args.family_beta):.3f} "
          f"= fitness {ev.fitness:.4g} + {args.family_beta} x weakest family "
          f"+ {args.family_beta} x weakest strength band (see selection_score)")
    print(f"fitness  : {ev.fitness:.4g} "
          f"(GROUP-RELATIVE advantage {ev.raw_fitness:+.2f} = (reward - the FINAL "
          f"round's cell mean) / its spread, where the cell also holds the "
          f"demand-gravity heuristic and reposition-OFF, averaged; "
          f"fallback {ev.fallback_rate:.2f}, defer {ev.defer_rate:.2f})")
    print(f"blindness: objective {ev.objective_blindness:.2f}, "
          f"fairness strength {ev.strength_blindness:.2f} [report only: 0 = the "
          f"fleet's target-region mix moves when that axis moves, 1 = never moves]")
    fam = " ".join(f"{k} {v:+.2f}" for k, v in sorted(ev.per_family.items()))
    print(f"per-family   advantage: {fam}")
    band = " ".join(f"{k} {v:+.2f}" for k, v in sorted(ev.per_strength.items()))
    print(f"per-strength advantage: {band}")
    if ev.per_family:
        print(f"weakest family : {min(ev.per_family, key=ev.per_family.get)} "
              f"({min(ev.per_family.values()):.2f})")
    if ev.per_strength:
        print(f"weakest band   : {min(ev.per_strength, key=ev.per_strength.get)} "
              f"({min(ev.per_strength.values()):.2f})")

    # The anchors come back HERE, on one FIXED batch: the only cross-round
    # comparable number, and the one the paper quotes.
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
        y_str = pair_by_strength_band(y_scen, y_obj, random.Random(args.seed + 99),
                                      _BANDS)
        card = reposition_yardstick(best, skills, combiner, y_scen, y_obj, y_str,
                                    combiner_code=combiner_code,
                                    fallback_penalty=args.fallback_penalty,
                                    workers=workers)
        print(f"yardstick: beats the demand-gravity heuristic on "
              f"{card['beats_heuristic']}/{card['n_cells']} fixed cells and "
              f"reposition-OFF on {card['beats_off']}/{card['n_cells']} "
              f"(mean advantage {card['rank']:+.2f}; mean reward "
              f"{card['champion_rewards']:.4g} vs heuristic "
              f"{card['anchor_rewards']['heuristic']:.4g} vs off "
              f"{card['anchor_rewards']['off']:.4g})")

    # The post-search soft check. Two axes, isolated the same way: the SAME hour,
    # seed and env reward are rolled with w withheld, with w handed over, and once
    # more with w handed over but the fairness strength forced to 0. Whatever
    # separates those columns is what reading each input bought. The objectives are
    # the model's own (the NL family) on TRAIN windows -- grading this on the
    # evaluation grid's hand-written objectives would be running the diagnostic on
    # the test set. Nothing feeds back: see pref_dispatch/llm/policy_audit.py.
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
        a_str = pair_by_strength_band(a_scen, a_obj,
                                      random.Random(args.seed + 555), _BANDS)
        cells = probe_phase3(GuardedScorer(best.scorer), combiner, skills,
                             a_scen, a_obj, a_str)
        best.meta["policy_audit"] = run_policy_audit(
            client, phase=3,
            objective=str(best.meta.get("objective", "")),
            description=str(best.meta.get("description", "")),
            code=str(best.meta.get("code", "")),
            cells=cells,
        )

    if args.freeze:
        path = freeze_repositioner(best)
        print(f"frozen   -> {path}")
    else:
        print("(not frozen: --no-freeze)")


if __name__ == "__main__":
    main()
