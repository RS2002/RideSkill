"""Feature-3 reposition-scorer evolution loop: propose -> improve -> freeze.

The order-scoring skill basis and upper combiner are FIXED input. This module
evolves the independent idle-driver reposition scorer -- the ONE thing the LLM
authors for Feature 3 (per-region base attractiveness). It mirrors
:func:`pref_dispatch.llm.evolve_pure_phase1.evolve_pure_phase1` (a compact
fixed-fitness hill-climb) but for the reposition surface:

1. **Generation 0 (propose).** Ask the model for a ``reposition_scores`` scorer:
   a chain-of-thought ``reposition_understanding`` first (interpretability gate),
   then name, objective, description, and the code. Compile + sandbox-validate
   against the synthetic region contract (int in-range keys, finite values).
2. **Generations 1..G (improve).** Show the model the current best scorer and its
   measured fixed fitness and ask it to score higher, keeping the contract. A
   (1+lambda) hill-climb: keep a candidate only if it validates AND its measured
   improvement-over-reposition-off beats the incumbent.
3. **Freeze.** Write the winning scorer to ``pref_dispatch/evolved/repositioners/``
   as a runnable ``reposition_scores`` module + a ``.meta.json`` with the CoT,
   description, provenance, and the fitness it was selected under.

The fitness is FIXED by the researcher (the ON-vs-OFF improvement over
repositioning being off on the same scenario, §7); the model never authors a
yardstick. Every reply flows through extract -> require NL explanation
(interpretability gate) -> sandbox. A reply that fails is fed back as
``repair_feedback`` for one retry. Live runs need ``YIBU_API_KEY`` (env only);
this module itself never touches a key.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from pref_dispatch.llm.client import LLMClient
from pref_dispatch.llm.combiner_eval import blindness_from_dists
# Shared with Phase 2 rather than re-typed: a frozen artifact is an experiment
# record, and ``evolved/`` is untracked, so a name the model happens to reuse must
# never overwrite an earlier one.
from pref_dispatch.llm.evolve_combiner import _unique_frozen_name
from pref_dispatch.llm.extract import extract_json, require_explanation
from pref_dispatch.llm.fitness_eval import (
    RepositionerEval,
    RepositionerRewardEval,
    evaluate_repositioner,
    evaluate_repositioner_reward_batch,
)
from pref_dispatch.llm.group_fitness import (
    DEFAULT_FALLBACK_PENALTY,
    DEFAULT_FAMILY_BETA,
)
from pref_dispatch.llm.parallel import (
    NotParallelizable,
    frozen_combiner_payload,
    parallel_reposition_anchors,
    parallel_reposition_rows,
)
from pref_dispatch.llm.prompts.reposition_evolve import (
    REQUIRED_EXPLANATION_FIELDS,
    REQUIRED_FIELDS,
    build_reposition_prompt,
)
# Repair policy (cooling schedule + unparseable-payload dumps) is shared with
# Phases 1 and 2 -- see pref_dispatch.llm.repair. Patch
# ``pref_dispatch.llm.repair.UNPARSEABLE_DIR`` to redirect the dumps; rebinding a
# name here would silently stop working.
from pref_dispatch.llm.repair import (
    REPAIR_MIN_TEMPERATURE as _REPAIR_MIN_TEMPERATURE,
    client_reply_header,
    dump_unparseable as _dump_unparseable,
    repair_temperature,
)
from pref_dispatch.llm.reposition_adapter import GuardedScorer
from pref_dispatch.llm.reposition_eval import (
    RepositionEval,
    _roll_cell_rewards,
    anchor_reference_rewards,
    group_evals,
    strength_label,
)
from pref_dispatch.llm.sandbox import (
    CompiledRepositioner,
    SandboxError,
    compile_repositioner,
    validate_repositioner,
)
from pref_dispatch.scenario import Scenario
from pref_dispatch.skills import Skill

FrozenDir = os.path.join("pref_dispatch", "evolved", "repositioners")


class EvolutionError(RuntimeError):
    """Raised when a generation cannot produce any valid candidate."""


@dataclass
class RepositionerCandidate:
    """A validated reposition scorer plus its provenance and measured fitness."""

    meta: Dict  # skill_name, objective, description, reposition_understanding, code, gen
    scorer: CompiledRepositioner
    # The fixed-fitness arm stores a RepositionerEval here; the v8 group loop stores
    # a RepositionEval. Both carry ``.fitness``, which is all ``score_value`` needs.
    evaluation: Optional[object] = None
    # Set once per round by the group loop: the GuardedScorer that THIS round's
    # rollouts actually ran through, so its telemetry and captured decision
    # contexts belong to the round being scored and not to an older batch.
    guarded: Optional[GuardedScorer] = None

    @property
    def name(self) -> str:
        return self.meta["skill_name"]

    @property
    def score_value(self) -> float:
        return self.evaluation.fitness if self.evaluation is not None else float("-inf")


def _require_w_read(code: str) -> None:
    """Reject a reposition scorer that never USES `w` to change its scoring.

    ``reposition_scores`` receives ``w`` (the episode objective). A scorer that
    reads ``w`` only to test ``is None`` and then returns the same rankings under
    every objective is COSMETICALLY objective-aware: it is charged by the runtime
    objective-blindness metric, but that only surfaces after rollouts. We catch it
    at GENERATION time instead: the code must actually CALL ``w`` (probe it on a
    small event) and then let that value influence the returned scores. A code that
    never calls ``w`` cannot respond to the objective whatever its prose claims.
    """
    if "w(" not in code:
        raise SandboxError(
            "repositioner never calls `w`: the objective is handed to you and you "
            "must read it. Probe `w` on a small event dict to learn what a "
            "finished trip / an extra minute of service / a detour is worth in "
            "THIS episode, and let that value change which regions you score "
            "highly -- otherwise you are objective-blind and every cell of the "
            "round is unanswerable."
        )


def _check_fields(obj: Dict) -> None:
    missing = [f for f in REQUIRED_FIELDS if not str(obj.get(f, "")).strip()]
    if missing:
        raise SandboxError(f"response missing required field(s): {missing}")
    require_explanation(obj, REQUIRED_EXPLANATION_FIELDS)


def _build_candidate(obj: Dict, gen: int) -> RepositionerCandidate:
    """Compile + validate a model response object into a candidate."""
    _check_fields(obj)
    name = str(obj["skill_name"]).strip()
    scorer = compile_repositioner(obj["code"], name=name)
    ok, why = validate_repositioner(scorer)
    if not ok:
        raise SandboxError(f"repositioner invalid: {why}")
    _require_w_read(obj["code"])
    meta = {
        "skill_name": name,
        "objective": obj["objective"].strip(),
        "objective_read_check": obj["objective_read_check"].strip(),
        "description": obj["description"].strip(),
        "reposition_understanding": obj["reposition_understanding"].strip(),
        "code": obj["code"],
        "gen": gen,
    }
    return RepositionerCandidate(meta=meta, scorer=scorer)


def _ask(client: LLMClient, prompt: Dict[str, str], temperature=None) -> Dict:
    raw = client.complete(prompt["system"], prompt["user"], temperature=temperature)
    try:
        return extract_json(raw)
    except Exception:
        _dump_unparseable(
            raw,
            header=(f"{client_reply_header(client)} "
                    f"chars={len(raw) if isinstance(raw, str) else -1}"),
        )
        raise


def _propose_with_repair(
    client: LLMClient,
    build_prompt: Callable[[Optional[str]], Dict[str, str]],
    gen: int,
    *,
    n_repair: int = 3,
    temperature: Optional[float] = None,
    log: Callable[[str], None] = print,
) -> RepositionerCandidate:
    """Call the model; on a validation error, feed it back and retry COOLER.

    Same policy as Phase 2: each retry drops the temperature (floored at
    :data:`_REPAIR_MIN_TEMPERATURE`), because malformed output is exactly the
    failure mode high temperature causes -- diversity is worth paying for in the
    FIRST attempt, not in the recovery. An unparseable reply is dumped to disk by
    :func:`_ask` before it is retried, so a generation that dies can be diagnosed
    from the actual payload instead of from the exception text.
    """
    feedback: Optional[str] = None
    last_err = ""
    for attempt in range(n_repair + 1):
        temp = repair_temperature(temperature, attempt)
        prompt = build_prompt(feedback)
        try:
            obj = _ask(client, prompt, temperature=temp)
            if attempt:
                log(f"    [repair] attempt {attempt + 1} succeeded at temp={temp}")
            return _build_candidate(obj, gen)
        except Exception as e:  # noqa: BLE001 -- retry on any
            last_err = f"{type(e).__name__}: {e}"
            feedback = last_err
            if attempt < n_repair:
                log(f"    [repair] attempt {attempt + 1} failed ({last_err}); "
                    f"retrying cooler")
    raise EvolutionError(f"no valid repositioner after repairs; last error: {last_err}")


def evolve_repositioner(
    client: LLMClient,
    env_profile: str,
    scenario: Scenario,
    *,
    generations: int = 4,
    lam: int = 2,
    temperature: float = 0.9,
    log: Callable[[str], None] = print,
) -> RepositionerCandidate:
    """Evolve ONE reposition scorer against the FIXED Feature-3 fitness.

    ``scenario`` pins the single operating point every candidate is measured on
    (same env seed, so ON-vs-OFF is an honest Δ). The fitness is fixed at every
    generation (including gen 0): the improvement in
    ``100*service_rate - mean_service_time - 5*reposition_distance_ratio`` that
    the scorer's repositioning brings over repositioning being OFF -- the model
    never authors a yardstick, it only maximises the given one.
    """
    def _eval(cand: RepositionerCandidate) -> RepositionerEval:
        return evaluate_repositioner(
            cand.scorer.reposition_scores, scenario,
        )

    # --- Generation 0: propose (explain targets, then scorer). ------------- #
    best = _propose_with_repair(
        client,
        lambda fb: build_reposition_prompt(env_profile, repair_feedback=fb),
        gen=0, temperature=temperature,
    )
    best.evaluation = _eval(best)
    log(
        f"[gen 0] {best.name!r} objective={best.meta['objective']!r} "
        f"fitness(ON-OFF)={best.evaluation.fitness:.4g}"
    )
    log(
        f"        deltas: service_rate {best.evaluation.delta_service_rate:+.4f}, "
        f"mean_service_time {best.evaluation.delta_mean_service_time:+.4f}, "
        f"reposition_ratio {best.evaluation.delta_reposition_ratio:+.4f}"
    )
    log(f"        reposition_understanding: {best.meta['reposition_understanding']}")

    # --- Generations 1..G: improve the scorer under the fixed fitness. ----- #
    for gen in range(1, generations + 1):
        for _ in range(lam):
            try:
                cand = _propose_with_repair(
                    client,
                    lambda fb: build_reposition_prompt(
                        env_profile,
                        current_code=best.meta["code"],
                        current_fitness=best.evaluation.fitness,
                        repair_feedback=fb,
                    ),
                    gen=gen, temperature=temperature,
                )
            except EvolutionError as e:
                log(f"[gen {gen}] candidate failed: {e}")
                continue
            cand.evaluation = _eval(cand)
            improved = cand.evaluation.fitness > best.evaluation.fitness
            log(
                f"[gen {gen}] {cand.name!r} fitness={cand.evaluation.fitness:.4g} "
                f"(service {cand.evaluation.delta_service_rate:+.4f}) "
                f"({'ACCEPT' if improved else 'reject'})"
            )
            if improved:
                best = cand

    return best


# =========================================================================== #
# Final version (B3): repositioner across random objectives AND fairness.       #
# =========================================================================== #


@dataclass
class RepositionBatchEval:
    """Mean fixed-fitness Δ of a reposition scorer over an objective+strength batch."""

    fitness: float                          # mean per-pair (ON - OFF) fixed-fitness Δ
    per_pair: Dict[int, float] = field(default_factory=dict)
    mean_delta_service_rate: float = 0.0
    mean_delta_mean_service_time: float = 0.0
    mean_delta_reposition_ratio: float = 0.0
    # One representative per-pair eval kept for freeze-time provenance (ON/OFF metrics).
    sample_eval: Optional[RepositionerEval] = None


def evaluate_repositioner_batch(
    scores_fn,
    scenarios: Sequence[Scenario],
    objectives: Sequence[object],
    strengths: Sequence[float],
    *,
    combiner: Optional[object] = None,
    skills: Optional[Dict[str, Skill]] = None,
) -> RepositionBatchEval:
    """Mean fixed reposition fitness over ``zip(scenarios, objectives, strengths)``.

    Each pair rolls the scorer ON vs OFF on one scenario, UNDER that pair's episode
    objective ``w`` (injected + handed to the scorer) AND that pair's fairness
    ``strength`` (the budget aggressiveness the scorer must best-respond to). The Δ
    is the FIXED Feature-3 fitness ``100*service_rate - mean_service_time -
    5*reposition_distance_ratio`` (ON - OFF) -- unchanged; ``w``/``strength`` only
    steer the scorer, they do not grade it. The mean over the batch is the fitness,
    so one frozen scorer is selected to best-respond ACROSS objectives and strengths
    (strength is part of its context, so a single artifact covers the axis)."""
    if not (len(scenarios) == len(objectives) == len(strengths)):
        raise ValueError("scenarios, objectives, strengths must be the same length")
    per_pair: Dict[int, float] = {}
    dsr: List[float] = []
    dmst: List[float] = []
    drr: List[float] = []
    sample_eval: Optional[RepositionerEval] = None
    for i, (sc, obj, strength) in enumerate(zip(scenarios, objectives, strengths)):
        ev = evaluate_repositioner(
            scores_fn, sc,
            combiner=combiner, skills=skills,
            strength=float(strength),
            reward_function=getattr(obj, "reward_function", None),
        )
        per_pair[i] = ev.fitness
        dsr.append(ev.delta_service_rate)
        dmst.append(ev.delta_mean_service_time)
        drr.append(ev.delta_reposition_ratio)
        if sample_eval is None:
            sample_eval = ev
    n = len(per_pair) or 1
    return RepositionBatchEval(
        fitness=sum(per_pair.values()) / n,
        per_pair=per_pair,
        mean_delta_service_rate=sum(dsr) / n,
        mean_delta_mean_service_time=sum(dmst) / n,
        mean_delta_reposition_ratio=sum(drr) / n,
        sample_eval=sample_eval,
    )


def evolve_repositioner_objectives(
    client: LLMClient,
    env_profile: str,
    scenarios: Sequence[Scenario],
    objectives: Sequence[object],
    strengths: Sequence[float],
    *,
    combiner: Optional[object] = None,
    skills: Optional[Dict[str, Skill]] = None,
    generations: int = 4,
    lam: int = 2,
    temperature: float = 0.9,
    log: Callable[[str], None] = print,
) -> RepositionerCandidate:
    """Evolve ONE reposition scorer across random objectives AND fairness strengths.

    The final-version Phase-3 loop (§B3). Skills + combiner are FROZEN. Every
    candidate is scored by :func:`evaluate_repositioner_reward_batch` over
    ``zip(scenarios, objectives, strengths)`` -- a spread of scenes, episode
    objectives (each handed to the scorer as ``w``) and fairness strengths (the
    budget aggressiveness in the scorer's context). The fitness is the SAME yardstick
    as Phases 1-2: the scorer's fleet-mean cumulative authored reward
    (``income_mean``), min-max normalised in each pair's single-skill reference frame
    and averaged. Maximising it selects ONE frozen scorer that lifts the OBJECTIVE
    (whatever it is) ACROSS objectives and strengths without retraining, matching how
    the budget mechanism already parameterises fairness.

    The prompt is unchanged: the scorer already receives ``w`` in its contract
    (:data:`~pref_dispatch.llm.prompts.reposition_evolve.REPOSITION_SIGNATURE_SPEC`);
    the randomization lives entirely in the evaluation batch. Per-pair reference
    frames are built ONCE (via
    :func:`pref_dispatch.llm.combiner_eval.build_objective_frames`) and shared across
    candidates in the run, so the ruler is fixed and the LLM cannot game it."""
    if not (len(scenarios) == len(objectives) == len(strengths)):
        raise ValueError("scenarios, objectives, strengths must be the same length")

    # Resolve the frozen dispatch stack once (also needed to build the frames).
    if combiner is None or skills is None:
        from pref_dispatch.llm.basis import load_basis, load_frozen_combiner

        _skills, _cards = load_basis(include_evolved=True)
        _combiner, _ = load_frozen_combiner(skill_names=tuple(_skills))
        if combiner is None:
            combiner = _combiner
        if skills is None:
            skills = _skills

    # FIXED per-(scenario, objective) env-reward frames: the scale-free ruler that
    # makes income_mean comparable across the batch's wildly different reward scales.
    # Built once from single-skill reference rollouts (the LLM never sees / governs it).
    from pref_dispatch.llm.combiner_eval import build_objective_frames

    reward_fns = [getattr(o, "reward_function", None) for o in objectives]
    frames = build_objective_frames(skills, scenarios, reward_fns)

    def _eval(cand: RepositionerCandidate) -> RepositionerRewardEval:
        return evaluate_repositioner_reward_batch(
            cand.scorer.reposition_scores, scenarios, objectives, strengths, frames,
            combiner=combiner, skills=skills,
        )

    n_blind = sum(1 for o in objectives if getattr(o, "reward_function", None) is None)
    log(
        f"[B3] evolving over {len(scenarios)} (scene,objective,strength) pairs "
        f"({n_blind} objective-blind); reward-under-w fitness (Phase-2 yardstick); "
        f"strengths={[round(float(s), 2) for s in strengths]}"
    )

    best = _propose_with_repair(
        client,
        lambda fb: build_reposition_prompt(env_profile, repair_feedback=fb),
        gen=0, temperature=temperature,
    )
    batch = _eval(best)
    best._reward_eval = batch                  # type: ignore[attr-defined]
    best._batch_fitness = batch.fitness        # type: ignore[attr-defined]
    log(
        f"[gen 0] {best.name!r} objective={best.meta['objective']!r} "
        f"mean fitness(norm reward)={batch.fitness:.4g} "
        f"(mean income_mean {batch.mean_income_mean:.4g})"
    )
    log(f"        reposition_understanding: {best.meta['reposition_understanding']}")

    for gen in range(1, generations + 1):
        for _ in range(lam):
            try:
                cand = _propose_with_repair(
                    client,
                    lambda fb: build_reposition_prompt(
                        env_profile,
                        current_code=best.meta["code"],
                        current_fitness=best._batch_fitness,  # type: ignore[attr-defined]
                        repair_feedback=fb,
                    ),
                    gen=gen, temperature=temperature,
                )
            except EvolutionError as e:
                log(f"[gen {gen}] candidate failed: {e}")
                continue
            cbatch = _eval(cand)
            cand._reward_eval = cbatch  # type: ignore[attr-defined]
            cand._batch_fitness = cbatch.fitness  # type: ignore[attr-defined]
            improved = cbatch.fitness > best._batch_fitness  # type: ignore[attr-defined]
            log(
                f"[gen {gen}] {cand.name!r} mean fitness={cbatch.fitness:.4g} "
                f"(income_mean {cbatch.mean_income_mean:.4g}) "
                f"({'ACCEPT' if improved else 'reject'})"
            )
            if improved:
                best = cand

    return best


# =========================================================================== #
# Final version (v8): (mu+lambda) group-relative evolution, Phase-2 parity.    #
# =========================================================================== #
# Phase 3 is Phase 2's loop with ONE extra axis. Everything structural is the
# same on purpose -- (mu+lambda), scenes rotating every round, parents re-rolled
# with the offspring, behavioural clone kill, one targeted runtime repair then
# elimination, crossover, parentless injection, per-round checkpoints -- because
# those were the v6/v7 changes that took Phase 2 from 12/30 to 27/30 and there is
# no reason the repositioner's search would want a different one. The extra axis
# is the FAIRNESS STRENGTH: a cell is (scene, objective, strength), and selection
# charges the weakest strength band exactly as it charges the weakest objective
# family.


def selection_score(ev: RepositionEval, *, beta: float = DEFAULT_FAMILY_BETA) -> float:
    """Selection key: mean gain over doing nothing, plus ``beta`` x weakest family/band.

    With the user-required default ``beta = 0`` (2026-08-13) the key is exactly
    the pure GRPO mean advantage; the reserved per-family and per-strength-band
    elite slots in :func:`select_survivors` keep the best specialist on each
    dimension alive as crossover material without weighting any of them in the key.

    ``ev.fitness`` is ``ev.raw_fitness`` unless a caller re-enables the legacy
    fallback penalty. It no longer needs one: a crash parks the car (see
    :meth:`GuardedScorer._stay`), so a program that raises everywhere IS the
    do-nothing baseline and scores 0 by construction -- it cannot buy anything
    back here because there is nothing borrowed to give back.
    """
    fam = min(ev.per_family.values(), default=0.0)
    band = min(ev.per_strength.values(), default=0.0)
    return ev.fitness + beta * fam + beta * band


def _seed_repositioner_candidate(seed_code: str, seed_meta: Dict) -> RepositionerCandidate:
    """Warm-start incumbent: compile a FROZEN scorer's code as a gen-0 candidate.

    No LLM call. Missing NL fields fall back to placeholders so the contract check
    passes -- the code is the payload here, the provenance comes from ``seed_meta``.
    """
    obj = {
        "skill_name": str(seed_meta.get("skill_name", "seed_repositioner")).strip()
        or "seed_repositioner",
        "objective": str(seed_meta.get("objective", "warm-start seed")).strip()
        or "warm-start seed",
        "description": str(seed_meta.get(
            "description", "Frozen repositioner seeded for further evolution.")).strip()
        or "Frozen repositioner seeded for further evolution.",
        "reposition_understanding": str(seed_meta.get(
            "reposition_understanding", "Inherited from the frozen seed scorer.")).strip()
        or "Inherited from the frozen seed scorer.",
        "code": seed_code,
    }
    return _build_candidate(obj, gen=0)


def select_survivors(
    pool: Sequence[RepositionerCandidate],
    objectives: Sequence[object],
    strengths: Sequence[float],
    *,
    mu: int = 4,
    family_beta: float = DEFAULT_FAMILY_BETA,
    log: Callable[[str], None] = lambda _s: None,
) -> List[RepositionerCandidate]:
    """The ``mu`` best by :func:`selection_score`, plus family AND band elites.

    With the user-required default ``family_beta = 0`` the key is exactly the pure
    GRPO mean advantage. Two reserved-slot passes on top of the plain top-``mu``
    cut: the pool member with the highest ``per_family[f]`` survives for every
    objective family, and the one with the highest ``per_strength[b]`` survives
    for every fairness band present in the round. A scorer that is the only thing
    anyone has that still works at ``strong`` is exactly the crossover material a
    strong all-rounder needs, and the plain mean key would otherwise cull it.
    """
    def _sel(c: RepositionerCandidate) -> float:
        return selection_score(c.evaluation, beta=family_beta) \
            if c.evaluation is not None else float("-inf")

    ranked = sorted(pool, key=_sel, reverse=True)
    keep: List[RepositionerCandidate] = list(ranked[:max(1, mu)])
    kept = {id(c) for c in keep}

    def _reserve(axis: str, key: str) -> None:
        scored = [c for c in pool
                  if c.evaluation is not None
                  and key in getattr(c.evaluation, axis)]
        if not scored:
            return
        champ = max(scored, key=lambda c: getattr(c.evaluation, axis)[key])
        if id(champ) in kept:
            return
        keep.append(champ)
        kept.add(id(champ))
        log(f"    [elite] {champ.name!r} survives on {key} "
            f"({getattr(champ.evaluation, axis)[key]:.2f}) despite selection "
            f"{_sel(champ):.3f}")

    for fam in sorted({getattr(o, "family", "?") for o in objectives}):
        _reserve("per_family", fam)
    for band in sorted({strength_label(s) for s in strengths}):
        _reserve("per_strength", band)
    return keep


def evolve_repositioner_group(
    client: LLMClient,
    env_profile: str,
    frozen_skills: Dict[str, Skill],
    combiner,
    scenarios: Sequence[Scenario],
    objectives: Sequence[object],
    strengths: Sequence[float],
    *,
    combiner_code: Optional[str] = None,
    batch_fn: Optional[Callable[[int], "tuple"]] = None,
    seed_code: Optional[str] = None,
    seed_meta: Optional[Dict] = None,
    generations: int = 4,
    mu: int = 4,
    lam: int = 4,
    crossover_rate: float = 0.35,
    fresh_per_round: int = 1,
    rng: Optional[random.Random] = None,
    temperature: float = 0.9,
    fallback_penalty: float = 0.0,
    family_beta: float = DEFAULT_FAMILY_BETA,
    capture: int = 400,
    workers: int = 1,
    checkpoint_fn: Optional[Callable[[int, "RepositionerCandidate"], None]] = None,
    patience: int = 0,
    min_gen: int = 0,
    runoff: bool = False,
    log: Callable[[str], None] = print,
) -> RepositionerCandidate:
    """Evolve ONE reposition scorer with the Phase-2 ``(mu+lambda)`` loop.

    Skills and combiner are FROZEN input. Every candidate is scored on the round's
    cells -- ``zip(scenarios, objectives, strengths)`` -- where each cell injects
    its own reward into the env, hands it to the scorer as ``w``, and sets the
    matcher's fairness budget to its own strength.

    **Fitness is group-relative (GRPO) within a cell**: a candidate's score on a
    cell is ``(r - mean)/std`` over the programs alive that round PLUS the two
    fixed anchors -- the built-in demand-gravity heuristic and repositioning
    switched OFF. The anchors are what make the number mean something absolute: a
    round in which every program is worse than no repositioning at all still
    produces advantages near 0 without them, and the whole point of Phase 3 is that
    moving idle cars must beat not moving them. The mean over cells is the raw
    fitness; only the fallback penalty is subtracted.

    Structurally identical to :func:`~pref_dispatch.llm.evolve_combiner.evolve_combiner_objectives`
    -- scenes rotate per round via ``batch_fn(round) -> (scenarios, objectives,
    strengths)``, parents are re-rolled with the offspring so the round's
    comparison is exact and paired, behaviourally identical programs are killed
    (first in pool order survives), a program that fell back at runtime gets ONE
    repair carrying the real cause and is eliminated if it still falls back,
    offspring are an LLM mutation of one survivor or (at ``crossover_rate``) an LLM
    crossover of two, and ``fresh_per_round`` of the ``lam`` slots are parentless
    injections so a genuinely new mechanism can still enter after round 0.

    The ONE difference is the fairness axis: selection charges the weakest strength
    band as well as the weakest objective family (:func:`selection_score`), an
    elite slot is reserved per band, and the report carries a second blindness
    number -- whether the fleet's target-region mix moves at all when the budget
    goes from off to dominating.

    ``combiner_code`` is the frozen combiner's ``skill_scores`` source; it is only
    needed for ``workers > 1`` (a compiled sandbox function cannot be pickled, so
    every worker rebuilds the stack from source). Without it the parallel path
    reports itself unusable and the round runs in-process.

    ``checkpoint_fn(generation, leader)`` fires after every round including gen 0.
    Returns the best of the FINAL round -- advantages from different rounds are
    measured on different scenes and are not comparable.
    """
    if not (len(scenarios) == len(objectives) == len(strengths)):
        raise ValueError("scenarios, objectives and strengths must be the same length")
    if mu < 1 or lam < 1:
        raise ValueError("mu and lam must both be >= 1")
    rnd = rng or random.Random(0)
    comb_payload = None
    if workers > 1:
        try:
            comb_payload = frozen_combiner_payload(combiner_code, tuple(frozen_skills))
        except NotParallelizable as e:
            log(f"    [parallel] combiner cannot be shipped ({e}); running in-process")
            workers = 1

    def _batch(round_idx: int):
        """This round's cells; the fixed batch when no sampler is given."""
        if batch_fn is None:
            return list(scenarios), list(objectives), list(strengths)
        scs, objs, sts = batch_fn(round_idx)
        if not (len(scs) == len(objs) == len(sts)):
            raise ValueError(f"round {round_idx}: batch_fn returned {len(scs)} "
                             f"scenarios, {len(objs)} objectives, {len(sts)} strengths")
        return list(scs), list(objs), list(sts)

    def _sel(c: RepositionerCandidate) -> float:
        return selection_score(c.evaluation, beta=family_beta) \
            if c.evaluation is not None else float("-inf")

    def _anchors(scs, objs, sts):
        """The heuristic / reposition-off references for this round's cells."""
        if workers > 1:
            try:
                return parallel_reposition_anchors(
                    comb_payload, frozen_skills, scs, objs, sts,
                    workers=workers, log=log)
            except NotParallelizable as e:
                log(f"    [parallel] anchors not usable ({e}); running in-process")
        return anchor_reference_rewards(combiner, frozen_skills, scs, objs, sts)

    def _repair_runtime(cand: RepositionerCandidate, gen: int
                        ) -> Optional[RepositionerCandidate]:
        """One targeted fix for a scorer that broke at RUNTIME.

        Sandbox validation only proves the code parses and returns in-range region
        keys on a synthetic driver; a scorer can still raise on the ten-thousandth
        idle car because one observation key is missing in a scarce-fleet scene.
        That is a bug, not a cost, so it costs exactly one LLM call with the real
        cause attached -- and the program is dropped if the fix does not take.
        """
        g = cand.guarded
        why = (getattr(g, "first_fallback_reason", None) or "unknown cause")
        note = (f"RUNTIME FAILURE: your reposition_scores raised or returned nothing "
                f"usable on {g.fallback_rate:.1%} of idle-driver decisions, so those "
                f"cars were PARKED WHERE THEY STOOD -- a crash buys you the score of "
                f"doing nothing on those drivers, which is exactly 0. "
                f"First cause: {why}. Fix THAT -- read every observation field with "
                f".get(key, default), never assume a key exists, and always return a "
                f"finite float for every region index you emit. Returning an EMPTY "
                f"dict on purpose is fine (it defers to the built-in demand-gravity "
                f"heuristic and is not charged); raising is not. Keep the strategy "
                f"identical.")
        try:
            fixed = _propose_with_repair(
                client,
                lambda fb: build_reposition_prompt(
                    env_profile,
                    current_code=cand.meta["code"],
                    current_fitness=0.0,
                    current_fitness_note=note,
                    repair_feedback=fb or note,
                ),
                gen=gen,
                temperature=min(temperature, _REPAIR_MIN_TEMPERATURE + 0.2),
                log=log,
            )
        except EvolutionError as e:
            log(f"    [runtime-repair] {cand.name!r} unrecoverable: {e}")
            return None
        fixed.meta["operator"] = f"runtime-repair({cand.name})"
        fixed.meta["parents"] = [cand.name]
        return fixed

    def _roll_rows(pool: Sequence[RepositionerCandidate], scs, objs, sts):
        """Roll every candidate on every cell; return ``(rows, blind, sblind)``.

        On one core this is the plain loop and the blindness probes are replayed
        locally off each scorer's own capture. With ``workers > 1`` the rollouts run
        in worker processes; the telemetry and both probes are measured THERE (the
        capture buffer lives with the rollout) and copied back, so everything
        downstream reads the same fields either way.
        """
        for c in pool:
            c.guarded = GuardedScorer(c.scorer)
        if workers > 1 and pool:
            try:
                recs = parallel_reposition_rows(
                    pool, comb_payload, frozen_skills, scs, objs, sts,
                    workers=workers, capture=capture, log=log)
            except NotParallelizable as e:
                log(f"    [parallel] not usable this round ({e}); running in-process")
            else:
                blind, sblind = [], []
                for c, rec in zip(pool, recs):
                    c.guarded.n_calls = rec["n_calls"]
                    c.guarded.n_fallbacks = rec["n_fallbacks"]
                    c.guarded.n_defers = rec["n_defers"]
                    c.guarded.first_fallback_reason = rec["reason"]
                    blind.append(blindness_from_dists(rec["picks"] or []))
                    sblind.append(blindness_from_dists(rec["spicks"] or []))
                return [rec["rewards"] for rec in recs], blind, sblind
        rows = [_roll_cell_rewards(c.guarded, combiner, frozen_skills,
                                   scs, objs, sts, capture=capture)
                for c in pool]
        return rows, None, None

    def _score_pool(pool: List[RepositionerCandidate], scs, objs, sts,
                    refs, gen: int) -> set:
        """Roll EVERY member of ``pool`` on this round's cells and rank the group.

        Parents and offspring alike -- that is what makes the comparison paired.
        Each candidate gets a fresh :class:`GuardedScorer`, so its fallback rate and
        its captured decision contexts belong to THIS round.

        Returns the set of eliminated ``id()``s: behavioural clones (identical
        reward row -- keep the first in pool order, which is a surviving parent
        before a child that merely reproduced it) and programs still falling back
        after their one repair.
        """
        rows, blind, sblind = _roll_rows(pool, scs, objs, sts)
        broken = set()
        seen_rows: Dict[tuple, RepositionerCandidate] = {}
        for c, row in zip(pool, rows):
            key = tuple(row)
            first = seen_rows.get(key)
            if first is None:
                seen_rows[key] = c
                continue
            broken.add(id(c))
            log(f"    [clone] {c.name!r} is behaviourally IDENTICAL to "
                f"{first.name!r} on all {len(row)} cells -- ELIMINATED this round")
        for i, c in enumerate(list(pool)):
            if c.guarded.fallback_rate <= 0.0:
                continue
            log(f"    [runtime] {c.name!r} broke on "
                f"{c.guarded.fallback_rate:.1%} of idle-driver decisions "
                f"({c.guarded.first_fallback_reason}) -- one repair attempt")
            fixed = _repair_runtime(c, gen)
            if fixed is not None:
                one_row, one_blind, one_sblind = _roll_rows([fixed], scs, objs, sts)
                rows[i] = one_row[0]
                if blind is not None and one_blind is not None:
                    blind[i] = one_blind[0]
                if sblind is not None and one_sblind is not None:
                    sblind[i] = one_sblind[0]
                pool[i] = fixed
                c = fixed
            if c.guarded.fallback_rate > 0.0:
                broken.add(id(c))
                log(f"    [runtime] {c.name!r} still breaks "
                    f"({c.guarded.fallback_rate:.1%}) -- ELIMINATED this round")
        for row in rows:
            assert len(row) == len(scs), (
                f"rollout row {len(row)} does not match the round's grid {len(scs)}")
        evals = group_evals(rows, refs, objs, sts,
                            scorers=[c.guarded for c in pool],
                            fallback_penalty=fallback_penalty,
                            blindness=blind, strength_blindness=sblind)
        for c, ev in zip(pool, evals):
            c.evaluation = ev
        return broken

    def _survivors(pool, objs, sts, broken) -> List[RepositionerCandidate]:
        """Top-``mu`` plus elites; eliminated programs are excluded unless that
        would empty the archive (then the round has nothing better and they stay)."""
        alive = [c for c in pool if id(c) not in broken] or list(pool)
        return select_survivors(alive, objs, sts, mu=mu,
                                family_beta=family_beta, log=log)

    def _fitness_note(cand: RepositionerCandidate) -> str:
        ev = cand.evaluation
        fam = " ".join(f"{k} {v:.2f}" for k, v in sorted(ev.per_family.items()))
        band = " ".join(f"{k} {v:.2f}" for k, v in sorted(ev.per_strength.items()))
        worst_f = min(ev.per_family, key=ev.per_family.get) if ev.per_family else "?"
        worst_b = min(ev.per_strength, key=ev.per_strength.get) if ev.per_strength else "?"
        mn_f = min(ev.per_family.values(), default=0.0)
        mn_b = min(ev.per_strength.values(), default=0.0)
        # Degeneracy read-out. Under the delta fitness a 0.00 is no longer
        # ambiguous about the BASELINE (it always means "worth the same as doing
        # nothing"), but it is still ambiguous about the ROUND: a whole column
        # that never diverged is a search that has stalled, and the LLM has to be
        # told that outright or it reads 0.00 as an ordinary middling score.
        dead = [f for f, t in sorted(ev.family_tie_rate.items()) if t >= 0.5]
        deg = ""
        if dead:
            bits = []
            for f in dead:
                above = ev.family_anchors_above.get(f, 0.0)
                bits.append(f"{f} (identical to {ev.family_tie_rate[f]:.0%} of the "
                            f"round; {above:.1f} of the 2 fixed references -- the "
                            f"demand-gravity heuristic and repositioning OFF -- "
                            f"BEAT you there)")
            deg = (" DEAD FAMILIES -- your fleet produced the SAME episode reward as "
                   "the rest of the round on: " + "; ".join(bits) +
                   ". That means no program in this round, yours included, sent "
                   "cars anywhere different when the objective changed shape. Make "
                   "the target region VISIBLY depend on what you are asked to "
                   "maximise.")
        return (
            f"SELECTION {_sel(cand):.3f} = mean gain-over-doing-nothing "
            f"{ev.raw_fitness:.2f} (pure GRPO advantage; no family or band term -- "
            f"beta is 0). per-family: {fam or 'n/a'}; "
            f"per-strength: {band or 'n/a'}. YOUR WEAKEST FAMILY: {worst_f} "
            f"{mn_f:.2f}; YOUR WEAKEST FAIRNESS BAND: {worst_b} {mn_b:.2f} -- a "
            f"specialist survives via the reserved per-family/per-band slots, not "
            f"via any weight on the weakest one. Your score on a cell is (YOUR "
            f"episode reward - the reward of NOT REPOSITIONING AT ALL on the SAME "
            f"scene and seed) / how much this round's programs disagree about that "
            f"same difference. So reward SCALES do NOT matter, and the sign is "
            f"absolute: 0.00 means you were worth exactly as much as leaving every "
            f"car parked, NEGATIVE means sending cars around ACTIVELY LOST money, "
            f"and +1 means your gain over doing nothing is one whole spread above "
            f"how much the field's gains vary. The built-in demand-gravity "
            f"heuristic is in that field as a landmark. Beating the other "
            f"candidates is NOT the target -- beating 'do nothing' is. broke on "
            f"{ev.fallback_rate:.2f} of decisions (a raise PARKS that car, so "
            f"crashing scores 0, never the heuristic's result), deferred on "
            f"{ev.defer_rate:.2f}. objective blindness {ev.objective_blindness:.2f}, "
            f"fairness blindness {ev.strength_blindness:.2f} (0 = where you send "
            f"cars visibly moves when the objective / the fairness budget changes, "
            f"1 = it never moves)." + deg
        )

    def _fmt_rank(ev: RepositionEval) -> str:
        fam = " ".join(f"{k} {v:.2f}" for k, v in sorted(ev.per_family.items()))
        band = " ".join(f"{k} {v:.2f}" for k, v in sorted(ev.per_strength.items()))
        tied = ",".join(f for f, t in sorted(ev.family_tie_rate.items()) if t >= 0.5)
        return (f"adv {ev.raw_fitness:+.2f} -> fitness {ev.fitness:.3f} "
                f"(selection {selection_score(ev, beta=family_beta):.3f}; "
                f"per-family: {fam or 'n/a'}; per-strength: {band or 'n/a'}, "
                f"broke {ev.fallback_rate:.2f}, deferred {ev.defer_rate:.2f}, "
                f"blindness obj {ev.objective_blindness:.2f} / fair "
                f"{ev.strength_blindness:.2f}"
                + (f", TIED-FAMILIES {tied}" if tied else "") + ")")

    def _parent_card(c: RepositionerCandidate) -> Dict:
        return {"name": c.name, "objective": c.meta.get("objective", ""),
                "code": c.meta["code"], "fitness_note": _fitness_note(c)}

    def _offspring(gen: int, survivors: Sequence[RepositionerCandidate]
                   ) -> Optional[RepositionerCandidate]:
        """One LLM child: crossover of two random survivors, else mutation of one."""
        do_cross = len(survivors) >= 2 and rnd.random() < crossover_rate
        if do_cross:
            pa, pb = rnd.sample(list(survivors), 2)
            kind = f"crossover({pa.name} x {pb.name})"
            cards = [_parent_card(pa), _parent_card(pb)]
            build = lambda fb: build_reposition_prompt(  # noqa: E731
                env_profile, parents=cards, repair_feedback=fb)
        else:
            pa = rnd.choice(list(survivors))
            kind = f"mutation({pa.name})"
            build = lambda fb: build_reposition_prompt(  # noqa: E731
                env_profile,
                current_code=pa.meta["code"],
                current_fitness=pa.evaluation.fitness,
                current_fitness_note=_fitness_note(pa),
                repair_feedback=fb)
        try:
            cand = _propose_with_repair(client, build, gen=gen,
                                        temperature=temperature, log=log)
        except EvolutionError as e:
            log(f"[gen {gen}] {kind} failed: {e}")
            return None
        cand.meta["operator"] = kind
        cand.meta["parents"] = ([pa.name, pb.name] if do_cross else [pa.name])
        return cand

    def _fresh_child(gen: int) -> Optional[RepositionerCandidate]:
        """One PARENTLESS scorer, proposed from the task alone.

        Every mutation/crossover child descends from a gen-0 program, so without
        this slot the whole archive converges to one template wearing different
        thresholds. It competes on the same footing -- if it is worse than the
        archive it simply does not survive -- so the cost is one rollout row.
        """
        try:
            cand = _propose_with_repair(
                client,
                lambda fb: build_reposition_prompt(env_profile, repair_feedback=fb),
                gen=gen, temperature=temperature, log=log,
            )
        except EvolutionError as e:
            log(f"[gen {gen}] fresh (parentless) proposal failed: {e}")
            return None
        cand.meta["operator"] = "fresh(parentless)"
        cand.meta["parents"] = []
        return cand

    log(f"[B3e] (mu={mu}+lambda={lam}) group-relative evolution, {generations} "
        f"round(s) x {len(objectives)} (scene, objective, strength) cells; scenes "
        f"{'ROTATE per round' if batch_fn else 'FIXED across rounds'}; anchors = "
        f"demand-gravity heuristic + reposition OFF; crossover rate {crossover_rate}")

    # --- Round 0: fill the archive. ---------------------------------------- #
    scs, objs, sts = _batch(0)
    refs = _anchors(scs, objs, sts)
    pool: List[RepositionerCandidate] = []
    if seed_code is not None:
        pool.append(_seed_repositioner_candidate(seed_code, seed_meta or {}))
    while len(pool) < mu:
        try:
            pool.append(_propose_with_repair(
                client,
                lambda fb: build_reposition_prompt(env_profile, repair_feedback=fb),
                gen=0, temperature=temperature, log=log,
            ))
        except EvolutionError as e:
            log(f"[gen 0] proposal failed: {e}")
            if not pool:
                raise
            break
    broken = _score_pool(pool, scs, objs, sts, refs, 0)
    for c in sorted(pool, key=_sel, reverse=True):
        log(f"[gen 0] {c.name!r} objective={c.meta['objective']!r} "
            f"{_fmt_rank(c.evaluation)}"
            f"{' [ELIMINATED]' if id(c) in broken else ''}")
    archive = _survivors(pool, objs, sts, broken)
    if checkpoint_fn is not None:
        checkpoint_fn(0, max(archive, key=_sel))

    # Round leaders for the runoff final (dedup by code) + patience streak.
    # Same mechanism as evolve_combiner_objectives (2026-08-13).
    round_leaders: List[RepositionerCandidate] = []
    leader_codes: set = set()

    def _remember_leader(champ: "RepositionerCandidate") -> None:
        code = str(champ.meta.get("code", ""))
        if code and code not in leader_codes:
            leader_codes.add(code)
            round_leaders.append(champ)

    _remember_leader(max(archive, key=_sel))
    streak_code: Optional[str] = str(max(archive, key=_sel).meta.get("code", ""))
    streak = 1

    # --- Rounds 1..G: mutate/cross, re-roll EVERYTHING, reselect. ---------- #
    for gen in range(1, generations + 1):
        scs, objs, sts = _batch(gen)
        refs = _anchors(scs, objs, sts)
        n_fresh = max(0, min(int(fresh_per_round), lam))
        raw_children = ([_fresh_child(gen) for _ in range(n_fresh)]
                        + [_offspring(gen, archive) for _ in range(lam - n_fresh)])
        children = [c for c in raw_children if c is not None]
        if not children:
            log(f"[gen {gen}] no valid offspring; archive re-rolled anyway")
        pool = list(archive) + children
        alive = {id(c) for c in archive}
        broken = _score_pool(pool, scs, objs, sts, refs, gen)
        for c in sorted(pool, key=_sel, reverse=True):
            tag = "parent" if id(c) in alive else c.meta.get("operator", "child")
            log(f"[gen {gen}] {c.name!r} [{tag}] {_fmt_rank(c.evaluation)}"
                f"{' [ELIMINATED]' if id(c) in broken else ''}")
        archive = _survivors(pool, objs, sts, broken)
        champ = max(archive, key=_sel)
        log(f"[gen {gen}] archive {[c.name for c in archive]}; "
            f"leader {champ.name!r} selection {_sel(champ):.3f}")
        if checkpoint_fn is not None:
            checkpoint_fn(gen, champ)
        _remember_leader(champ)

        champ_code = str(champ.meta.get("code", ""))
        streak = streak + 1 if champ_code == streak_code else 1
        streak_code = champ_code
        if patience > 0 and gen >= min_gen and streak >= patience:
            log(f"[stop] leader {champ.name!r} unchanged for {streak} consecutive "
                f"round(s) (patience {patience}) after round {gen} >= min_gen "
                f"{min_gen} -- search converged, stopping after round "
                f"{gen}/{generations}")
            break

    # --- Runoff final: all round leaders, one fresh batch, one paired GRPO. -- #
    if runoff and len(round_leaders) > 1 and batch_fn is not None:
        runoff_idx = generations + 1
        scs, objs, sts = _batch(runoff_idx)
        refs = _anchors(scs, objs, sts)
        log(f"[runoff] {len(round_leaders)} distinct round leader(s) re-rolled "
            f"together on a FRESH batch of {len(scs)} cell(s): "
            f"{[c.name for c in round_leaders]}")
        pool = list(round_leaders)
        broken = _score_pool(pool, scs, objs, sts, refs, runoff_idx)
        for c in sorted(pool, key=_sel, reverse=True):
            log(f"[runoff] {c.name!r} {_fmt_rank(c.evaluation)}"
                f"{' [ELIMINATED]' if id(c) in broken else ''}")
        contenders = [c for c in pool if id(c) not in broken] or pool
        winner = max(contenders, key=_sel)
        log(f"[runoff] winner {winner.name!r} selection {_sel(winner):.3f} "
            f"(pure same-batch GRPO; no cross-round comparison)")
        if checkpoint_fn is not None:
            checkpoint_fn(runoff_idx, winner)
        return winner
    if runoff and batch_fn is None:
        log("[runoff] skipped: no batch_fn to draw a fresh batch from")
    elif runoff:
        log("[runoff] skipped: only one distinct round leader")

    return max(archive, key=_sel)


def reposition_yardstick(
    champion: RepositionerCandidate,
    frozen_skills: Dict[str, Skill],
    combiner,
    scenarios: Sequence[Scenario],
    objectives: Sequence[object],
    strengths: Sequence[float],
    *,
    combiner_code: Optional[str] = None,
    fallback_penalty: float = 0.0,
    workers: int = 1,
    log: Callable[[str], None] = print,
) -> Dict[str, object]:
    """Score ``champion`` against the two fixed anchors on a FIXED batch.

    The training batches rotate, so training scores from different rounds are not
    comparable to each other. This one is: one fixed batch of cells, rolled once,
    giving the learning-curve point and the paper's headline -- how much of the
    time the evolved scorer beats the built-in demand-gravity heuristic, and how
    much of the time it beats not repositioning at all. That second number is the
    one that matters: a repositioner that loses to switching itself off is not a
    repositioner.

    Returns ``{"rank", "beats_heuristic", "beats_off", "n_cells", "per_family",
    "per_strength", "anchor_rewards", "champion_rewards"}``; ``rank`` is the mean
    standardised advantage against the anchors alone.
    """
    scorer = GuardedScorer(champion.scorer)
    blind = sblind = None
    if workers > 1:
        try:
            payload = frozen_combiner_payload(combiner_code, tuple(frozen_skills))
            refs = parallel_reposition_anchors(payload, frozen_skills, scenarios,
                                               objectives, strengths,
                                               workers=workers, log=log)
            recs = parallel_reposition_rows([champion], payload, frozen_skills,
                                            scenarios, objectives, strengths,
                                            workers=workers, log=log)
        except NotParallelizable as e:
            log(f"    [parallel] yardstick not usable ({e}); running in-process")
            workers = 1
        else:
            rec = recs[0]
            row = rec["rewards"]
            scorer.n_calls = rec["n_calls"]
            scorer.n_fallbacks = rec["n_fallbacks"]
            scorer.n_defers = rec["n_defers"]
            scorer.first_fallback_reason = rec["reason"]
            blind = [blindness_from_dists(rec["picks"] or [])]
            sblind = [blindness_from_dists(rec["spicks"] or [])]
    if workers <= 1:
        refs = anchor_reference_rewards(combiner, frozen_skills, scenarios,
                                        objectives, strengths)
        row = _roll_cell_rewards(scorer, combiner, frozen_skills, scenarios,
                                 objectives, strengths)
    evals = group_evals([row], refs, objectives, strengths, scorers=[scorer],
                        fallback_penalty=fallback_penalty,
                        blindness=blind, strength_blindness=sblind)
    ev = evals[0]
    n = len(row) or 1
    beats_heur = sum(1 for r, ref in zip(row, refs) if r > ref[0])
    beats_off = sum(1 for r, ref in zip(row, refs) if r > ref[1])
    log(f"[yardstick] {champion.name!r} mean advantage {ev.raw_fitness:+.2f} over "
        f"{n} fixed cells; beats the demand-gravity heuristic on {beats_heur}/{n} "
        f"and reposition-OFF on {beats_off}/{n}")
    return {
        "rank": ev.raw_fitness,
        "beats_heuristic": beats_heur,
        "beats_off": beats_off,
        "n_cells": n,
        "per_family": dict(ev.per_family),
        "per_strength": dict(ev.per_strength),
        "anchor_rewards": {
            "heuristic": sum(r[0] for r in refs) / max(1, len(refs)),
            "off": sum(r[1] for r in refs) / max(1, len(refs)),
        },
        "champion_rewards": sum(row) / n,
    }


def freeze_repositioner(
    cand: RepositionerCandidate,
    out_dir: str = FrozenDir,
) -> str:
    """Write the winning reposition scorer to disk as a runnable module + meta.

    The module defines exactly one function, ``reposition_scores`` (plus a
    primitive header); the loader recompiles the body through the sandbox.
    """
    os.makedirs(out_dir, exist_ok=True)
    name = _unique_frozen_name(cand.name, out_dir)
    py_path = os.path.join(out_dir, f"{name}.py")
    meta_path = os.path.join(out_dir, f"{name}.meta.json")

    ev = cand.evaluation
    group_ev = ev if isinstance(ev, RepositionEval) else None
    if group_ev is not None:
        fitness_line = (
            "Evolved under GROUP-RELATIVE fitness: on each (scene, objective,\n"
            "fairness-strength) cell, (my reward - the cell's mean) / the cell's\n"
            "spread, where the cell also holds the demand-gravity heuristic and\n"
            "repositioning switched OFF"
        )
    elif getattr(cand, "_reward_eval", None) is not None:
        fitness_line = (
            "Evolved under the reward-under-w fitness (normalised income_mean, the\n"
            "same yardstick as Phases 1-2), over a batch of sampled objectives and\n"
            "fairness strengths"
        )
    else:
        fitness_line = (
            "Evolved under the FIXED Feature-3 fitness (improvement over repositioning\n"
            "OFF on the same scenario)"
        )
    header = (
        f'"""Frozen Feature-3 reposition scorer: {name}\n\n'
        f"Objective: {cand.meta['objective']}\n\n"
        f"{cand.meta['description']}\n\n"
        f"Reposition understanding (LLM CoT): "
        f"{cand.meta['reposition_understanding']}\n\n"
        f"{fitness_line}, gen {cand.meta['gen']}. Authors ONLY per-region\n"
        f"base scores: coordinated spreading, stay rules, and the relocate action\n"
        f'stay in pref_dispatch.reposition."""\n\n'
        "import math\n"
        "import numpy as np\n\n\n"
    )
    with open(py_path, "w", encoding="utf-8") as f:
        f.write(header + cand.meta["code"].rstrip() + "\n")

    meta = dict(cand.meta)
    # Keep the recorded name identical to the file name, so a loader asked for
    # ``--repositioner-name`` finds it under the name it was actually written as.
    meta["skill_name"] = name
    if name != cand.name:
        meta["proposed_name"] = cand.name
    reward_eval = getattr(cand, "_reward_eval", None)
    if group_ev is not None:
        # v8 Phase-3: group-relative, with the fairness axis.
        meta["fitness"] = group_ev.fitness
        meta["raw_fitness"] = group_ev.raw_fitness
        meta["fallback_rate"] = group_ev.fallback_rate
        meta["defer_rate"] = group_ev.defer_rate
        meta["per_family"] = group_ev.per_family
        meta["per_strength"] = group_ev.per_strength
        meta["objective_blindness"] = group_ev.objective_blindness
        meta["strength_blindness"] = group_ev.strength_blindness
        meta["fitness_def"] = (
            "mean over (scenario, objective, fairness strength) cells of the "
            "standardised advantage (r - mean)/std of income_mean under that "
            "cell's authored reward; the group is the programs alive that round "
            "plus two fixed anchors (the demand-gravity heuristic and "
            "repositioning OFF). Selection adds beta x the weakest objective "
            "family and beta x the weakest fairness band."
        )
    elif reward_eval is not None:
        # Final-version B3: graded by the Phase-2 reward-under-w yardstick.
        meta["fitness"] = reward_eval.fitness
        meta["mean_income_mean"] = reward_eval.mean_income_mean
        meta["per_pair_score"] = reward_eval.per_pair
        if reward_eval.sample_metrics is not None:
            meta["sample_metrics"] = reward_eval.sample_metrics
        meta["fitness_def"] = (
            "mean over (scenario,objective,strength) of scalarize_reward(income_mean); "
            "income_mean min-max normalised in each pair's single-skill reference "
            "frame -- the SAME fitness as Phase 1/2 (reward under w)"
        )
    elif cand.evaluation is not None:
        # Original fixed-fitness arm (single-scenario ON vs OFF).
        meta["fitness"] = cand.evaluation.fitness
        meta["delta_service_rate"] = cand.evaluation.delta_service_rate
        meta["delta_mean_service_time"] = cand.evaluation.delta_mean_service_time
        meta["delta_reposition_ratio"] = cand.evaluation.delta_reposition_ratio
        meta["metrics_on"] = cand.evaluation.metrics_on
        meta["metrics_off"] = cand.evaluation.metrics_off
        meta["fixed_fitness"] = (
            "100*service_rate - mean_service_time - 5*reposition_distance_ratio; "
            "fitness = ON - OFF on the same scenario"
        )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return py_path
