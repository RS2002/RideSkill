"""Phase-1 skill evolution as a ``(mu+lambda)`` group-relative search.

This replaces the ``(1+lambda)`` hill-climb in :mod:`pref_dispatch.llm.evolve` for
the final-version run. The paradigm the user set is unchanged and is the reason
this module exists as its own loop:

    **One skill is trained ALONE, under its OWN fixed LLM-authored objective.**
    Generation 0 chooses the objective and writes the ``fitness(metrics)`` that
    grades it; everything after that may only rewrite the ``score``/``noop_score``
    body. Different skills are never mixed into one population -- a population here
    is ``mu + lambda`` VARIANTS of a single skill, all graded by that skill's own
    fitness. Mixing them would make the fitness meaningless (each skill's fitness is
    self-authored and not comparable to any other's) and would collapse the
    repository toward a universal maximiser, which is exactly what Phase 1 must not
    produce.

What is group-relative, then, if the group is one skill's variants
-----------------------------------------------------------------
The GRPO normalisation runs **within each scenario**, across the round's variants::

    A_ik = (raw_ik - mean_k(raw_.k)) / std_k(raw_.k)

for variant ``i`` on scenario ``k``, then the variant's fitness is the mean of its
``A_ik`` over the round's scenarios. This is the fix for the problem Phase 1 has
had all along: a self-authored fitness has an arbitrary internal scale AND its raw
value swings with the scenario (a 1400-car peak hour produces far bigger numbers
than a 200-car off-peak one), so averaging raw fitness across a mixed batch is
mostly a vote for whichever scenes happen to be big. Standardising inside each
scenario removes both -- the mean absorbs the scene's level, the std its spread --
and what survives is only "how far above this round's other variants was I, here".

A useful consequence: with the per-scenario baseline built INTO the number, the old
``rescale="reference"`` mode (and its extra reference rollout per scenario) is no
longer needed. The round's own population is the baseline.

Scale is the diversity axis here (there are no objective families)
------------------------------------------------------------------
Phase 2 groups a candidate's advantages by OBJECTIVE FAMILY so a program that is
strong on average and hopeless on one family cannot hide behind its mean. Phase 1
has exactly one objective, so its equivalent axis is FLEET SCALE
(:func:`~pref_dispatch.llm.batch_pairing.band_label`): selection is
``mean advantage + band_beta * (weakest band's advantage)`` and each fleet band
additionally reserves one elite slot. A skill that only works at 1200 cars is not
what the repository wants, and the gate's hard cells are all at scarcity.

Skills have no fallback mechanism
---------------------------------
A combiner that raises is caught and silently falls back to a default skill, which
is why Phase 2 measures a fallback RATE. :class:`~pref_dispatch.llm.sandbox.CompiledSkill`
has no such net -- :mod:`pref_dispatch.matching` calls ``sk.score(...)`` bare, so an
exception propagates out of the rollout and kills the episode. The analogue here is
therefore an INVALID rate: the share of the round's scenarios on which the variant
raised. It gets one targeted repair carrying the real exception, and is eliminated
for the round if it still raises. Same policy, different failure surface.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from pref_dispatch.llm.batch_pairing import DEFAULT_FLEET_BANDS, band_label
from pref_dispatch.llm.client import LLMClient
from pref_dispatch.llm.evolve import (
    Candidate,
    EvolutionError,
    _propose_with_repair,
)
from pref_dispatch.llm.fitness_eval import rollout_skill_on_scenario
from pref_dispatch.llm.group_fitness import (
    DEFAULT_FALLBACK_PENALTY,
    DEFAULT_FAMILY_BETA,
    group_advantages,
    tie_rate,
)
from pref_dispatch.llm.prompts.skill_evolve import (
    build_skill_improve_prompt,
    build_skill_prompt,
)
from pref_dispatch.llm.repair import REPAIR_MIN_TEMPERATURE


@dataclass
class SkillGroupEval:
    """One variant's group-relative result over a round's scenario batch.

    ``raw_fitness`` is the mean standardised advantage; ``fitness`` subtracts the
    invalid-rollout penalty. ``per_scenario_raw`` keeps the pre-standardisation
    fitness values (``nan`` where the rollout raised) so a run can be re-scored
    offline without re-rolling anything.
    """

    fitness: float
    raw_fitness: float
    per_scenario_adv: List[float] = field(default_factory=list)
    per_scenario_raw: List[float] = field(default_factory=list)
    per_scenario_metrics: List[Optional[Dict[str, float]]] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    per_band: Dict[str, float] = field(default_factory=dict)
    band_tie_rate: Dict[str, float] = field(default_factory=dict)
    invalid_rate: float = 0.0
    # First exception a rollout raised, verbatim -- the repair prompt's payload.
    error: Optional[str] = None


def selection_score(ev: Optional[SkillGroupEval],
                    beta: float = DEFAULT_FAMILY_BETA) -> float:
    """``fitness + beta * (weakest band's advantage)`` -- the ranking key.

    With the user-required default ``beta = 0`` (2026-08-13) the key is exactly
    the pure GRPO mean advantage; the reserved per-band elite slots in
    :func:`select_survivors` keep the best specialist on each fleet band alive as
    crossover material without weighting any band in the key.
    """
    if ev is None:
        return float("-inf")
    weakest = min(ev.per_band.values(), default=0.0)
    return ev.fitness + beta * weakest


def group_evals(
    rows_raw: Sequence[Sequence[float]],
    rows_metrics: Sequence[Sequence[Optional[Dict[str, float]]]],
    scenarios: Sequence[object],
    *,
    errors: Optional[Sequence[Optional[str]]] = None,
    invalid_penalty: float = DEFAULT_FALLBACK_PENALTY,
    bands: Sequence[Tuple[int, int]] = DEFAULT_FLEET_BANDS,
) -> List[SkillGroupEval]:
    """Score a whole round's population against each other, scenario by scenario.

    ``rows_raw[i][k]`` is variant ``i``'s self-authored fitness on scenario ``k``,
    or a non-finite value where that rollout raised. Each SCENARIO COLUMN is
    standardised across the population (:func:`group_advantages`), which is the
    within-scenario GRPO the user specified -- never across scenarios, where the
    numbers are not comparable.

    A raising rollout is punished TWICE, and both are needed. Inside its column the
    non-finite value already floors at ``-Z_CLIP``, which is what makes a crash lose
    to any variant that merely did badly. But when EVERY variant raises on a scene
    the column has no finite members at all, so :func:`group_advantages` hands out
    0.0 across the board and that scene becomes indistinguishable from a healthy tie
    -- ``invalid_penalty * invalid_rate`` is the only term that still registers it.
    It is also the only part of the score that is comparable across rounds.
    """
    n_pop = len(rows_raw)
    n_sc = len(scenarios)
    labels = [getattr(sc, "label", lambda: "?")() for sc in scenarios]
    band_of = [band_label(sc, bands) for sc in scenarios]

    # Column-wise standardisation: for each scenario, the group is the whole
    # population's result on THAT scenario (candidate included -- that is the GRPO
    # definition and what makes a column's advantages sum to zero).
    cols_adv: List[List[float]] = []
    cols_raw: List[List[float]] = []
    for k in range(n_sc):
        col = [float(rows_raw[i][k]) if k < len(rows_raw[i]) else float("nan")
               for i in range(n_pop)]
        cols_raw.append(col)
        cols_adv.append(group_advantages(col))

    evals: List[SkillGroupEval] = []
    for i in range(n_pop):
        adv = [cols_adv[k][i] for k in range(n_sc)]
        raw = [cols_raw[k][i] for k in range(n_sc)]
        mets = [rows_metrics[i][k] if k < len(rows_metrics[i]) else None
                for k in range(n_sc)]

        by_band: Dict[str, List[float]] = {}
        tie_by_band: Dict[str, List[float]] = {}
        for k in range(n_sc):
            by_band.setdefault(band_of[k], []).append(adv[k])
            peers = [cols_raw[k][j] for j in range(n_pop) if j != i]
            tie_by_band.setdefault(band_of[k], []).append(tie_rate(cols_raw[k][i], peers))

        n_bad = sum(1 for x in raw if not math.isfinite(x))
        invalid_rate = n_bad / n_sc if n_sc else 0.0
        raw_fitness = sum(adv) / n_sc if n_sc else 0.0
        evals.append(SkillGroupEval(
            fitness=raw_fitness - invalid_penalty * invalid_rate,
            raw_fitness=raw_fitness,
            per_scenario_adv=adv,
            per_scenario_raw=raw,
            per_scenario_metrics=mets,
            labels=labels,
            per_band={b: sum(v) / len(v) for b, v in by_band.items()},
            band_tie_rate={b: sum(v) / len(v) for b, v in tie_by_band.items()},
            invalid_rate=invalid_rate,
            error=(errors[i] if errors is not None and i < len(errors) else None),
        ))
    return evals


def select_survivors(
    pool: Sequence[Candidate],
    *,
    mu: int = 3,
    band_beta: float = DEFAULT_FAMILY_BETA,
    log: Callable[[str], None] = lambda _s: None,
) -> List[Candidate]:
    """The ``mu`` best by :func:`selection_score`, plus one elite per fleet band.

    With the user-required default ``band_beta = 0`` the key is exactly the pure
    GRPO mean advantage. The reserved band slots exist because the plain mean key
    still culls a variant that is the best thing anyone has at ONE scale but
    unremarkable elsewhere -- and that variant is precisely the crossover material
    a strong all-rounder needs to stop being weak there.
    """
    def _sel(c: Candidate) -> float:
        return selection_score(c.evaluation, beta=band_beta)

    ranked = sorted(pool, key=_sel, reverse=True)
    keep: List[Candidate] = list(ranked[:max(1, mu)])
    kept = {id(c) for c in keep}
    bands = sorted({b for c in pool if isinstance(c.evaluation, SkillGroupEval)
                    for b in c.evaluation.per_band})
    for band in bands:
        scored = [c for c in pool if isinstance(c.evaluation, SkillGroupEval)
                  and band in c.evaluation.per_band]
        if not scored:
            continue
        champ = max(scored, key=lambda c: c.evaluation.per_band[band])
        if id(champ) not in kept:
            keep.append(champ)
            kept.add(id(champ))
            log(f"    [elite] {champ.name!r} survives on {band} "
                f"({champ.evaluation.per_band[band]:.2f}) despite selection "
                f"{_sel(champ):.3f}")
    return keep


def evolve_skill_group(
    client: LLMClient,
    env_profile: str,
    *,
    objective_hint: Optional[str] = None,
    existing_skills: Optional[Sequence[Dict]] = None,
    similarity_note: Optional[str] = None,
    repository_note: Optional[str] = None,
    audit_feedback: Optional[str] = None,
    scenarios: Sequence[object],
    batch_fn: Optional[Callable[[int], Sequence[object]]] = None,
    generations: int = 5,
    mu: int = 3,
    lam: int = 4,
    crossover_rate: float = 0.35,
    fresh_per_round: int = 1,
    rng: Optional[random.Random] = None,
    temperature: float = 0.9,
    invalid_penalty: float = DEFAULT_FALLBACK_PENALTY,
    band_beta: float = DEFAULT_FAMILY_BETA,
    bands: Sequence[Tuple[int, int]] = DEFAULT_FLEET_BANDS,
    workers: int = 1,
    checkpoint_fn: Optional[Callable[[int, Candidate], None]] = None,
    patience: int = 0,
    min_gen: int = 0,
    runoff: bool = False,
    log: Callable[[str], None] = print,
) -> Candidate:
    """Evolve ONE skill with ``(mu+lambda)`` + within-scenario GRPO. Returns the champion.

    Generation 0 makes one full :func:`build_skill_prompt` call -- that reply fixes
    the skill's NAME, OBJECTIVE and ``fitness_code`` for the entire run. The other
    ``mu - 1`` founders and every later child are variants of that ONE skill under
    that ONE fitness (see the module docstring).

    Rounds 1..``generations`` draw a fresh scenario batch from ``batch_fn(round)``
    when given (scenes ROTATE, so a variant cannot win by fitting one hour), build
    ``lam`` children -- ``fresh_per_round`` of them PARENTLESS, the rest mutation or
    crossover of survivors -- then re-roll parents AND children on the round's batch
    so every comparison is paired. Survivors are the top ``mu`` by
    :func:`selection_score` plus one elite per fleet band.

    ``checkpoint_fn(round_idx, leader)`` is called after each round's selection, so a
    run that dies at generation 7 does not lose generations 0-6.

    ``audit_feedback`` is set only when a previous COMPLETED search under this same
    brief was rejected by :func:`pref_dispatch.llm.skill_audit.audit_skill` for
    authoring a fitness that does not ask for the intended behaviour; it reaches
    generation 0 (the only generation that writes a fitness) and nothing else.

    ``workers > 1`` spreads the round's ``(mu + lambda) x len(batch)`` rollouts over
    processes; every rollout is seeded by its own scenario so the scores do not
    depend on how the work was split, and anything that cannot be shipped falls back
    to the in-process loop rather than half-running the round.
    """
    if mu < 1 or lam < 1:
        raise ValueError("mu and lam must both be >= 1")
    rnd = rng or random.Random(0)

    def _batch(round_idx: int) -> List[object]:
        return list(batch_fn(round_idx)) if batch_fn is not None else list(scenarios)

    def _sel(c: Candidate) -> float:
        return selection_score(c.evaluation, beta=band_beta)

    # ------------------------------------------------------------------ #
    # Rollouts.                                                          #
    # ------------------------------------------------------------------ #
    def _roll_one(cand: Candidate, scs: Sequence[object]):
        """One variant over the batch -> (raw fitness row, metrics row, first error).

        A scenario the skill raises on yields ``nan`` and NOT a zero: zero is a
        legitimate fitness value, and scoring a crash as "average" would let a skill
        that dies on scarce fleets outrank one that merely does badly there.
        """
        raw: List[float] = []
        mets: List[Optional[Dict[str, float]]] = []
        err: Optional[str] = None
        for sc in scs:
            try:
                m = rollout_skill_on_scenario(cand.skill, sc)
                v = float(cand.fitness_fn(m))
            except Exception as e:  # noqa: BLE001 -- a raising skill is a candidate too
                if err is None:
                    err = f"{type(e).__name__}: {e} (on {getattr(sc, 'label', lambda: '?')()})"
                raw.append(float("nan"))
                mets.append(None)
                continue
            raw.append(v if math.isfinite(v) else float("nan"))
            mets.append(m)
        return raw, mets, err

    def _roll_rows(pool: Sequence[Candidate], scs: Sequence[object]):
        """Roll every variant on every scenario -> (raw rows, metrics rows, errors)."""
        if workers > 1 and pool:
            try:
                from pref_dispatch.llm.parallel import (  # noqa: PLC0415
                    NotParallelizable,
                    parallel_skill_group_rows,
                )
            except ImportError as e:
                log(f"    [parallel] entry not available ({e}); running in-process")
            else:
                try:
                    return parallel_skill_group_rows(
                        pool, scs, workers=workers, log=log)
                except NotParallelizable as e:
                    log(f"    [parallel] not usable this round ({e}); "
                        f"running in-process")
        rows_raw, rows_met, errs = [], [], []
        for c in pool:
            raw, mets, err = _roll_one(c, scs)
            rows_raw.append(raw)
            rows_met.append(mets)
            errs.append(err)
        return rows_raw, rows_met, errs

    # ------------------------------------------------------------------ #
    # Read-outs handed back to the model.                                #
    # ------------------------------------------------------------------ #
    def _fitness_note(cand: Candidate) -> str:
        ev = cand.evaluation
        if not isinstance(ev, SkillGroupEval):
            return ""
        per = " ".join(f"{b} {v:+.2f}" for b, v in sorted(ev.per_band.items()))
        worst = min(ev.per_band, key=ev.per_band.get) if ev.per_band else "?"
        mn = ev.per_band.get(worst, 0.0)
        # Degeneracy read-out. A bare +0.00 is ambiguous: "middle of a spread field"
        # and "every variant produced the identical episode" print the same number,
        # and the second is a dead round that looks healthy in the log.
        dead = [b for b, t in sorted(ev.band_tie_rate.items()) if t >= 0.5]
        deg = ""
        if dead:
            deg = (" DEAD SCALES -- on " + ", ".join(dead) + " your skill produced "
                   "the SAME episode result as the rest of the round. An advantage "
                   "of ~0.00 there is NOT 'average': it means nothing anyone tried "
                   "this round, including you, changed the outcome at that fleet "
                   "size. Make the rule visibly do something different there.")
        bad = ""
        if ev.invalid_rate > 0:
            bad = (f" WARNING: your code RAISED on {ev.invalid_rate:.0%} of this "
                   f"round's scenes ({ev.error}); a skill that raises kills the "
                   f"episode outright -- there is no fallback.")
        return (
            f"SELECTION {_sel(cand):+.3f} = mean advantage {ev.raw_fitness:+.2f} "
            f"(invalid-penalised {ev.fitness:+.2f}) + {band_beta} x weakest scale. "
            f"per-scale: {per or 'n/a'}; YOUR WEAKEST SCALE: {worst} {mn:+.2f} -- "
            f"raising it is the fastest way to get selected. Your score on a scene "
            f"is (your fitness - the round's mean) / the round's spread, so the "
            f"fitness function's SCALE does not matter -- only how far you are from "
            f"the OTHER variants alive this round, in units of how spread out they "
            f"are. 0 = exactly average, +1 = one whole spread above the field."
            + deg + bad
        )

    def _fmt_rank(ev: Optional[SkillGroupEval]) -> str:
        if not isinstance(ev, SkillGroupEval):
            return "unscored"
        per = " ".join(f"{b} {v:+.2f}" for b, v in sorted(ev.per_band.items()))
        tied = ",".join(b for b, t in sorted(ev.band_tie_rate.items()) if t >= 0.5)
        return (f"adv {ev.raw_fitness:+.2f} -> fitness {ev.fitness:+.3f} "
                f"(selection {selection_score(ev, beta=band_beta):+.3f}; "
                f"per-scale: {per or 'n/a'}, invalid {ev.invalid_rate:.2f}"
                + (f", TIED-SCALES {tied}" if tied else "") + ")")

    # ------------------------------------------------------------------ #
    # Generation 0: the objective and its fitness are chosen ONCE.       #
    # ------------------------------------------------------------------ #
    # Draw round 0's batch BEFORE logging it: batch_fn is a sampler, so calling it
    # twice would draw two different batches and the log would describe one that was
    # never rolled.
    scs = _batch(0)
    log(f"[B1] (mu={mu}+lambda={lam}) group-relative skill evolution, "
        f"{generations} round(s) x {len(scs)} scene(s); "
        f"scenes {'ROTATE per round' if batch_fn else 'FIXED across rounds'}; "
        f"crossover rate {crossover_rate}")

    founder = _propose_with_repair(
        client,
        lambda fb: build_skill_prompt(
            env_profile,
            objective_hint=objective_hint,
            existing_skills=existing_skills,
            similarity_note=similarity_note,
            repository_note=repository_note,
            audit_feedback=audit_feedback,
            repair_feedback=fb,
        ),
        gen=0, require_mechanism=True, require_self_check=True,
        temperature=temperature,
    )
    founder.meta["operator"] = "founder"
    founder.meta["parents"] = []
    fixed_objective = founder.meta["objective"]
    fixed_fitness_code = founder.meta["fitness_code"]
    fixed_fitness_fn = founder.fitness_fn
    log(f"[gen 0] objective FIXED: {fixed_objective}")
    log(f"[gen 0] fitness FIXED:\n{fixed_fitness_code}")

    def _variant(gen: int, *, parents: Optional[Sequence[Candidate]] = None,
                 current: Optional[Candidate] = None,
                 note: Optional[str] = None) -> Optional[Candidate]:
        """One child under the FIXED objective+fitness: crossover, mutation or fresh."""
        cards = [{"name": p.name,
                  "mechanism": p.meta.get("mechanism", ""),
                  "code": p.meta["code"],
                  "fitness_note": _fitness_note(p)} for p in (parents or [])]
        if cards:
            kind = f"crossover({cards[0]['name']} x {cards[1]['name']})"
        elif current is not None:
            kind = f"mutation({current.name})"
        else:
            kind = "fresh(parentless)"
        try:
            cand = _propose_with_repair(
                client,
                lambda fb: build_skill_improve_prompt(
                    env_profile,
                    objective=fixed_objective,
                    fitness_code=fixed_fitness_code,
                    current_code=(current.meta["code"] if current is not None else None),
                    current_fitness=(current.evaluation.fitness
                                     if current is not None
                                     and isinstance(current.evaluation, SkillGroupEval)
                                     else None),
                    fitness_note=note,
                    parents=cards or None,
                    existing_skills=existing_skills,
                    repair_feedback=fb,
                ),
                gen=gen,
                fixed_fitness_fn=fixed_fitness_fn,
                fixed_fitness_code=fixed_fitness_code,
                fixed_objective=fixed_objective,
                require_mechanism=True,
                temperature=temperature,
            )
        except EvolutionError as e:
            log(f"[gen {gen}] {kind} failed: {e}")
            return None
        cand.meta["operator"] = kind
        cand.meta["parents"] = [p.name for p in (parents or [])] or (
            [current.name] if current is not None else [])
        return cand

    def _repair_runtime(cand: Candidate, gen: int) -> Optional[Candidate]:
        """One targeted fix for a variant that RAISED during a rollout.

        Validation-time repair only proves the code parses and has the right shape;
        a skill can still raise on the ten-thousandth driver because one obs key is
        missing in a scarce-fleet scene. That is a bug, not a cost, so it is worth
        exactly one LLM call with the real exception attached.
        """
        ev = cand.evaluation
        why = (ev.error if isinstance(ev, SkillGroupEval) and ev.error
               else "unknown cause")
        note = (f"RUNTIME FAILURE: your score/noop_score RAISED on "
                f"{ev.invalid_rate:.0%} of this round's scenes, which kills the "
                f"whole episode -- a skill has NO fallback. First cause: {why}. "
                f"Fix THAT: read every obs/order field with .get(key, default), "
                f"never assume a key exists or that a list is non-empty, guard every "
                f"division, and always return a finite float. Keep the decision rule "
                f"identical.")
        try:
            fixed = _propose_with_repair(
                client,
                lambda fb: build_skill_improve_prompt(
                    env_profile,
                    objective=fixed_objective,
                    fitness_code=fixed_fitness_code,
                    current_code=cand.meta["code"],
                    fitness_note=note,
                    existing_skills=existing_skills,
                    repair_feedback=fb or note,
                ),
                gen=gen,
                fixed_fitness_fn=fixed_fitness_fn,
                fixed_fitness_code=fixed_fitness_code,
                fixed_objective=fixed_objective,
                require_mechanism=True,
                temperature=min(temperature, REPAIR_MIN_TEMPERATURE + 0.2),
            )
        except EvolutionError as e:
            log(f"    [runtime-repair] {cand.name!r} unrecoverable: {e}")
            return None
        fixed.meta["operator"] = f"runtime-repair({cand.name})"
        fixed.meta["parents"] = [cand.name]
        return fixed

    def _score_pool(pool: List[Candidate], scs: Sequence[object], gen: int) -> set:
        """Roll EVERY member on this round's batch and rank the group.

        Parents and children alike -- that is what makes the round's comparison a
        paired one, and it is also why a scene batch may rotate without corrupting
        anything: nothing is ever compared across rounds.

        Returns the set of eliminated ``id()``s: behavioural clones (after the
        first) and variants that still raise after their one repair.
        """
        rows_raw, rows_met, errs = _roll_rows(pool, scs)
        broken = set()

        # --- BEHAVIOURAL CLONE KILL --------------------------------------- #
        # A variant's row of per-scene fitness values IS its behavioural
        # fingerprint. Two variants with identical rows are one behaviour
        # occupying two of the round's slots, and -- worse -- they pad the
        # standardisation group with copies of themselves, which is what drags a
        # whole column's std toward zero and kills the round's signal. Keep the
        # FIRST occurrence in pool order (the archive comes first, so a surviving
        # parent always outranks a child that merely reproduced it).
        seen: Dict[tuple, Candidate] = {}
        for c, row in zip(pool, rows_raw):
            key = tuple(row)
            first = seen.get(key)
            if first is None:
                seen[key] = c
                continue
            broken.add(id(c))
            log(f"    [clone] {c.name!r} is behaviourally IDENTICAL to "
                f"{first.name!r} on all {len(row)} scenes -- ELIMINATED this round")

        # --- One targeted repair per raising variant ----------------------- #
        for i, c in enumerate(list(pool)):
            if errs[i] is None:
                continue
            n_bad = sum(1 for x in rows_raw[i] if not math.isfinite(x))
            log(f"    [runtime] {c.name!r} raised on {n_bad}/{len(scs)} scenes "
                f"({errs[i]}) -- one repair attempt")
            c.evaluation = SkillGroupEval(
                fitness=0.0, raw_fitness=0.0,
                invalid_rate=n_bad / len(scs) if scs else 0.0, error=errs[i])
            fixed = _repair_runtime(c, gen)
            if fixed is None:
                broken.add(id(c))
                continue
            one_raw, one_met, one_err = _roll_rows([fixed], scs)
            rows_raw[i], rows_met[i], errs[i] = one_raw[0], one_met[0], one_err[0]
            pool[i] = fixed
            if one_err[0] is not None:
                broken.add(id(fixed))
                log(f"    [runtime] {fixed.name!r} still raises "
                    f"({one_err[0]}) -- ELIMINATED this round")

        for row in rows_raw:
            assert len(row) == len(scs), (
                f"rollout row {len(row)} does not match the round's batch {len(scs)}")
        evals = group_evals(rows_raw, rows_met, scs, errors=errs,
                            invalid_penalty=invalid_penalty, bands=bands)
        for c, ev in zip(pool, evals):
            c.evaluation = ev
        return broken

    def _survivors(pool: Sequence[Candidate], broken: set) -> List[Candidate]:
        """Top ``mu`` + band elites, excluding eliminated variants.

        Unless excluding them would empty the archive -- then the round has nothing
        better to offer and they stay, so the run continues instead of aborting.
        """
        alive = [c for c in pool if id(c) not in broken] or list(pool)
        return select_survivors(alive, mu=mu, band_beta=band_beta, log=log)

    # --- Round 0: fill the archive with variants of the founder. ----------- #
    pool: List[Candidate] = [founder]
    while len(pool) < mu:
        child = _variant(0)  # parentless: a different mechanism for the same objective
        if child is None:
            if len(pool) >= 1:
                break
            raise EvolutionError("generation 0 produced no valid variant")
        pool.append(child)
    broken = _score_pool(pool, scs, 0)
    for c in sorted(pool, key=_sel, reverse=True):
        log(f"[gen 0] {c.name!r} mechanism={c.meta.get('mechanism', '?')!r} "
            f"{_fmt_rank(c.evaluation)}"
            f"{' [ELIMINATED]' if id(c) in broken else ''}")
    archive = _survivors(pool, broken)
    if checkpoint_fn is not None:
        checkpoint_fn(0, max(archive, key=_sel))

    # Round leaders for the runoff final (dedup by code) + patience streak.
    # Same mechanism as evolve_combiner_objectives (2026-08-13). Note the skill's
    # OBJECTIVE and fitness are fixed at gen 0, so every leader here is a variant
    # of one skill under one yardstick -- the runoff just removes the last-round
    # scene-draw lottery from picking WHICH variant is frozen.
    round_leaders: List[Candidate] = []
    leader_codes: set = set()

    def _remember_leader(champ: Candidate) -> None:
        code = str(champ.meta.get("code", ""))
        if code and code not in leader_codes:
            leader_codes.add(code)
            round_leaders.append(champ)

    _remember_leader(max(archive, key=_sel))
    streak_code: Optional[str] = str(max(archive, key=_sel).meta.get("code", ""))
    streak = 1

    # --- Rounds 1..G: mutate/cross, re-roll EVERYTHING, reselect. ---------- #
    for gen in range(1, generations + 1):
        scs = _batch(gen)
        n_fresh = max(0, min(int(fresh_per_round), lam))
        children: List[Candidate] = []
        for _ in range(n_fresh):
            child = _variant(gen)
            if child is not None:
                children.append(child)
        for _ in range(lam - n_fresh):
            if len(archive) >= 2 and rnd.random() < crossover_rate:
                pa, pb = rnd.sample(list(archive), 2)
                child = _variant(gen, parents=[pa, pb])
            else:
                pa = rnd.choice(list(archive))
                child = _variant(gen, current=pa, note=_fitness_note(pa))
            if child is not None:
                children.append(child)
        if not children:
            log(f"[gen {gen}] no valid offspring; archive re-rolled anyway")

        pool = list(archive) + children
        parents = {id(c) for c in archive}
        broken = _score_pool(pool, scs, gen)
        for c in sorted(pool, key=_sel, reverse=True):
            tag = "parent" if id(c) in parents else c.meta.get("operator", "child")
            log(f"[gen {gen}] {c.name!r} [{tag}] {_fmt_rank(c.evaluation)}"
                f"{' [ELIMINATED]' if id(c) in broken else ''}")
        archive = _survivors(pool, broken)
        champ = max(archive, key=_sel)
        log(f"[gen {gen}] archive {[c.name for c in archive]}; "
            f"leader {champ.name!r} selection {_sel(champ):+.3f}")
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
        scs = _batch(runoff_idx)
        log(f"[runoff] {len(round_leaders)} distinct round leader(s) re-rolled "
            f"together on a FRESH batch of {len(scs)} scene(s): "
            f"{[c.name for c in round_leaders]}")
        pool = list(round_leaders)
        broken = _score_pool(pool, scs, runoff_idx)
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
