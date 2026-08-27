"""Phase-2 combiner evolution loop (§5): propose -> improve -> freeze.

The frozen skill basis is fixed input. This module evolves the UPPER combiner:

1. **Generation 0 (propose).** Ask the model for a ``skill_scores`` combiner:
   name, strategy, description, and the code. Compile + sandbox-validate against
   the frozen skill names (out-of-basis keys are rejected and fed back).
2. **Generations 1..G (improve).** Show the model the current best combiner and
   its measured scalarised fitness on ``W_train`` and ask it to score higher,
   keeping the contract. A (1+lambda) hill-climb: keep the candidate only if it
   validates AND its preference-averaged fitness beats the incumbent.
3. **Freeze.** Write the winning combiner to ``pref_dispatch/evolved/combiners/``
   as a runnable ``skill_scores`` module + a ``.meta.json`` with strategy,
   description, provenance, and the fitness/W_train it was selected under.

The Phase-2 fitness is FIXED by the researcher (scalarise over W_train, §5.4);
the model never authors it. Every reply flows through extract -> require NL
explanation (interpretability gate) -> sandbox. A reply that fails is fed back as
``repair_feedback`` for one retry.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from pref_dispatch.llm.client import LLMClient
from pref_dispatch.llm.combiner_adapter import LLMCombiner
from pref_dispatch.llm.combiner_eval import (
    CombinerEval,
    NormRanges,
    _baseline_rewards,
    _group_evals,
    _roll_pair_rewards,
    _skill_reference_rewards,
    blindness_from_dists,
    evaluate_combiner,
    evaluate_combiner_scenarios,
    scenario_norm_frames,
)
from pref_dispatch.llm.combiner_eval import DEFAULT_REGIMES
from pref_dispatch.llm.extract import extract_json, require_explanation
from pref_dispatch.llm.fitness_eval import EVAL_NUM_DRIVERS, EVAL_ORDER_LIMIT
from pref_dispatch.llm.group_fitness import (
    DEFAULT_FALLBACK_PENALTY,
    DEFAULT_FAMILY_BETA,
)
from pref_dispatch.llm.parallel import (
    NotParallelizable,
    parallel_baseline_rows,
    parallel_pair_rewards,
    parallel_skill_rows,
)
from pref_dispatch.llm.prompts.combiner_evolve import (
    REQUIRED_EXPLANATION_FIELDS,
    REQUIRED_FIELDS,
    REWARD_REQUIRED_EXPLANATION_FIELDS,
    REWARD_REQUIRED_FIELDS,
    build_combiner_prompt,
)
from pref_dispatch.matching import DEFAULT_BLEND_K, DEFAULT_TOP_K
from pref_dispatch.llm.sandbox import (
    CompiledCombiner,
    SandboxError,
    compile_combiner,
    validate_combiner,
)
from pref_dispatch.preference import Preference
from pref_dispatch.scenario import Scenario
from pref_dispatch.skills import Skill

FrozenDir = os.path.join("pref_dispatch", "evolved", "combiners")


class EvolutionError(RuntimeError):
    """Raised when a generation cannot produce any valid candidate."""


@dataclass
class CombinerCandidate:
    """A validated combiner plus its provenance and measured fitness."""

    meta: Dict  # combiner_name, strategy, description, code, gen
    scorer: CompiledCombiner
    skill_names: Sequence[str]
    evaluation: Optional[CombinerEval] = None

    @property
    def name(self) -> str:
        return self.meta["combiner_name"]

    def make_combiner(self, *, soft: bool = False, blend_k: int = DEFAULT_BLEND_K) -> LLMCombiner:
        return LLMCombiner(
            self.scorer, self.skill_names, soft=soft, blend_k=blend_k
        )

    @property
    def score_value(self) -> float:
        return self.evaluation.fitness if self.evaluation is not None else float("-inf")


def selection_score(ev: CombinerEval, *, beta: float = DEFAULT_FAMILY_BETA) -> float:
    """Selection key = mean advantage, plus ``beta`` x weakest-family one.

    ``ev.fitness`` already subtracts the fallback penalty, so a crashing candidate
    stays far behind. With the user-required default ``beta = 0`` (2026-08-13) the
    key is EXACTLY the pure GRPO mean advantage -- no hand-set constant enters the
    selection. The reserved per-family elite slots in :func:`select_survivors`
    keep the best specialist on each hard family alive as crossover material
    WITHOUT weighting any family in the key.
    """
    mn = min(ev.per_family.values(), default=0.0)
    return ev.fitness + beta * mn


def _check_fields(obj: Dict, *, reward_mode: bool = False) -> None:
    """Require every contract field, and non-empty NL explanation fields.

    In ``reward_mode`` the reward-conditioned contract also requires the
    ``reward_understanding`` CoT field (the model must explain the reward first).
    """
    required = REWARD_REQUIRED_FIELDS if reward_mode else REQUIRED_FIELDS
    explain = REWARD_REQUIRED_EXPLANATION_FIELDS if reward_mode else REQUIRED_EXPLANATION_FIELDS
    missing = [f for f in required if not str(obj.get(f, "")).strip()]
    if missing:
        raise SandboxError(f"response missing required field(s): {missing}")
    require_explanation(obj, explain)


def _build_candidate(
    obj: Dict, gen: int, skill_names: Sequence[str], *, reward_mode: bool = False,
    independent_events: bool = False, probe_event_evolve: bool = False,
) -> CombinerCandidate:
    """Compile + validate a model response object into a candidate."""
    if independent_events:
        from benchmark.run_probe_event_evolve import rewrite_candidate_code
        rewrite_candidate_code(obj)  # mutate obj["code"] in place if needed
    _check_fields(obj, reward_mode=reward_mode)
    scorer = compile_combiner(obj["code"])
    ok, why = validate_combiner(scorer, tuple(skill_names))
    if not ok:
        raise SandboxError(f"combiner invalid: {why}")
    if probe_event_evolve:
        _require_probe_coverage(obj["code"])
    meta = {
        "combiner_name": str(obj["combiner_name"]).strip(),
        "strategy": obj["strategy"].strip(),
        "description": obj["description"].strip(),
        "code": obj["code"],
        "gen": gen,
    }
    # Carry the reward CoT into provenance when present (reward-conditioned arm).
    if str(obj.get("reward_understanding", "")).strip():
        meta["reward_understanding"] = obj["reward_understanding"].strip()
    return CombinerCandidate(meta=meta, scorer=scorer, skill_names=tuple(skill_names))


# Terms a probe-event combiner must be able to read, and a marker keyword each
# would plausibly appear as in the generated code. A probe set that never
# touches these cannot read a reward that prices them -- so we reject it and let
# the repair loop ask the model to add the missing probes.
#
# This is the SOFTWARE backstop for the prompt's "probe EVERY term you can
# imagine, even if this reward ignores it": the combiner must reference each of
# these axis-specific fields at least once so a reward pricing any of them can be
# read. Detour is split into the per-order pooled detour on a NEWLY assigned order
# (``assigned_detour_times``) and the aggregate re-routing impact on ONBOARD
# orders (``extra_detour_time``), because a reward may price one without the other
# and a probe that reads one cannot read the other.
#
# NOTE: volume (new-order count) and seating (party size) are NOT listed here.
# Their fields (``assigned_orders`` / ``assigned_party_sizes``) appear in every
# probe event by construction, so a substring check cannot tell whether the
# combiner actually probes those axes -- the MANDATORY PROBE RULES in the prompt
# carry that obligation instead. The fields below are axis-SPECIFIC: referencing
# them at all is meaningful evidence of a probe.
_REQUIRED_PROBE_MARKERS = (
    ("completion (drop-off)", "completed_orders"),
    ("dispatch_wait", "assigned_dispatch_wait"),
    ("pickup_time", "assigned_pickup_times"),
    ("detour on a NEW order", "assigned_detour_times"),
    ("detour on ONBOARD orders", "extra_detour_time"),
    ("solo/service time", "assigned_solo_times"),
    ("empty move", "is_empty_move"),
    ("idle wait", "is_idle_wait"),
)


def _require_probe_coverage(code: str) -> None:
    """Reject a probe combiner whose probes cannot read at least the KEY terms.

    We cannot parse intent, so we check that the generated code REFERENCE each
    required term's field at least once (any mention, in a probe event or a read).
    A code that never writes/reads ``assigned_pickup_times`` literally cannot
    probe pickup-time, whatever the prose claims. Detour is split into the
    NEW-order per-order pooled detour (``assigned_detour_times``) and the ONBOARD
    re-routing impact (``extra_detour_time``): a reward may price one and not the
    other, and probing one does not read the other.
    """
    missing = []
    for label, marker in _REQUIRED_PROBE_MARKERS:
        if marker not in code:
            missing.append(label)
    if missing:
        raise SandboxError(
            "probe coverage incomplete: the probe events never reference "
            f"{', '.join(missing)} -- a reward pricing any of these cannot be read. "
            "Add probe events for EVERY term (dispatch_wait, pickup_time, detour on "
            "a NEW order, detour on ONBOARD orders, solo/service time) even if you "
            "think this reward ignores them; an unused probe is harmless."
        )


def _seed_candidate(
    seed_code: str,
    seed_meta: Dict,
    skill_names: Sequence[str],
    *,
    reward_mode: bool = False,
) -> CombinerCandidate:
    """Warm-start incumbent: compile + validate a frozen combiner's code as gen-0.

    No LLM call. ``seed_meta`` supplies the name/strategy/description/provenance
    of the already-frozen combiner; ``seed_code`` is its ``skill_scores`` body.
    Missing NL fields fall back to placeholders so :func:`_build_candidate`'s
    contract check passes (the code itself is the real payload here).
    """
    obj = {
        "combiner_name": str(seed_meta.get("combiner_name", "seed_combiner")).strip()
        or "seed_combiner",
        "strategy": str(seed_meta.get("strategy", "warm-start seed")).strip()
        or "warm-start seed",
        "description": str(
            seed_meta.get("description", "Frozen combiner seeded for fine-tuning.")
        ).strip()
        or "Frozen combiner seeded for fine-tuning.",
        "code": seed_code,
    }
    ru = str(seed_meta.get("reward_understanding", "")).strip()
    if ru:
        obj["reward_understanding"] = ru
    elif reward_mode:
        obj["reward_understanding"] = "Inherited from the frozen seed combiner."
    return _build_candidate(obj, gen=0, skill_names=skill_names, reward_mode=reward_mode)


# Repair-pass temperature schedule + payload dumping moved to the shared module on
# 2026-08-10 so Phases 1 and 3 use the same policy. The two constants are re-exported
# under their old private names (they are read-only, so an alias is faithful).
# _UNPARSEABLE_DIR deliberately is NOT re-exported: it is the one name that used to
# be REBOUND to redirect the dumps, and an alias would silently stop working. Patch
# ``pref_dispatch.llm.repair.UNPARSEABLE_DIR`` instead.
from pref_dispatch.llm.repair import (
    REPAIR_COOLDOWN as _REPAIR_COOLDOWN,
    REPAIR_MIN_TEMPERATURE as _REPAIR_MIN_TEMPERATURE,
    client_reply_header,
    dump_unparseable as _dump_unparseable,
    repair_temperature,
)


def _ask(client: LLMClient, prompt: Dict[str, str], temperature=None) -> Dict:
    raw = client.complete(prompt["system"], prompt["user"], temperature=temperature)
    try:
        obj = extract_json(raw)
        # Some LLMs return a list of JSON objects; merge all dicts, last wins
        # on duplicate keys, to keep all fields (code, strategy, etc.)
        if isinstance(obj, list):
            merged: Dict = {}
            for item in obj:
                if isinstance(item, dict):
                    merged.update(item)
            if merged:
                return merged
            raise ExtractionError("extract_json returned a list with no dict entries")
        return obj
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
    skill_names: Sequence[str],
    *,
    reward_mode: bool = False,
    n_repair: int = 3,
    temperature: Optional[float] = None,
    log: Callable[[str], None] = print,
    independent_events: bool = False,
    probe_event_evolve: bool = False,
) -> CombinerCandidate:
    """Call the model; on a validation error, feed it back and retry COOLER.

    Each retry drops the temperature by :data:`_REPAIR_COOLDOWN` (floored at
    :data:`_REPAIR_MIN_TEMPERATURE`). Retrying at the SAME high temperature just
    re-rolls the same dice: three Phase-2 runs died at generation 0 with a JSON
    payload that failed to parse, all three at temperature 1.0, while the
    temperature-0.9 run succeeded. Malformed output is exactly the failure mode
    high temperature causes, so the repair pass should ask more conservatively --
    diversity is worth paying for in the FIRST attempt, not in the recovery."""
    feedback: Optional[str] = None
    last_err = ""
    for attempt in range(n_repair + 1):
        temp = repair_temperature(temperature, attempt)
        prompt = build_prompt(feedback)
        try:
            obj = _ask(client, prompt, temperature=temp)
            if attempt:
                log(f"    [repair] attempt {attempt + 1} succeeded at temp={temp}")
            return _build_candidate(obj, gen, skill_names, reward_mode=reward_mode,
                                    independent_events=independent_events,
                                    probe_event_evolve=probe_event_evolve)
        except Exception as e:  # noqa: BLE001 -- retry on any
            last_err = f"{type(e).__name__}: {e}"
            feedback = last_err
            if attempt < n_repair:
                log(f"    [repair] attempt {attempt + 1} failed ({last_err}); "
                    f"retrying cooler")
    raise EvolutionError(f"no valid combiner after repairs; last error: {last_err}")


def evolve_combiner(
    client: LLMClient,
    env_profile: str,
    frozen_skills: Dict[str, Skill],
    frozen_cards: Sequence[Dict],
    train_prefs: Sequence[Preference],
    ranges: NormRanges,
    *,
    scenarios: Optional[Sequence[Scenario]] = None,
    scenario_frames: Optional[Sequence[NormRanges]] = None,
    reward_spec: Optional[str] = None,
    reward_function=None,
    ignore_pref: bool = False,
    probe_event_evolve: bool = False,
    objective: str = "scalarize",
    seed_code: Optional[str] = None,
    seed_meta: Optional[Dict] = None,
    generations: int = 4,
    lam: int = 2,
    regimes: Sequence[str] = DEFAULT_REGIMES,
    split: str = "train",
    num_drivers: int = EVAL_NUM_DRIVERS,
    order_limit: Optional[int] = EVAL_ORDER_LIMIT,
    seed: int = 0,
    temperature: float = 0.9,
    log: Callable[[str], None] = print,
) -> CombinerCandidate:
    """Evolve the upper combiner and return the best candidate.

    ``frozen_skills`` maps name -> Skill (the frozen basis the combiner selects
    over); ``frozen_cards`` are their prompt cards.

    Two evaluation modes:

    * **v2 (``scenarios`` given).** The combiner is scored over a batch of
      domain-randomized scenarios (each with its own preference + scale-free
      per-scenario frame), so it must read BOTH the preference AND the scene. The
      per-scenario frames are built once (``scenario_frames``, or computed here)
      and shared across candidates for a fair comparison. ``ranges``/``train_prefs``
      are ignored in this mode.
    * **v1 (``scenarios`` None).** The legacy W_train x fixed-regime scalarisation
      against the fixed ``ranges`` frame. Kept for regression.

    §Phase-2 single-reward arm (``objective="env_reward"``, ``ignore_pref=True``,
    ``reward_spec`` + ``reward_function`` set): the combiner is composed for ONE
    fixed authored/given reward with NO runtime preference dial. ``reward_function``
    is injected into every scenario env so ``income_mean`` measures that reward's
    fleet-mean cumulative value (the fitness); ``ignore_pref`` switches the prompt
    to the no-preference contract; the smoothness penalty is auto-disabled under
    ``env_reward`` (see :func:`evaluate_combiner_scenarios`). Only the scenarios
    path supports reward injection.

    §Phase-3 warm-start fine-tune (``seed_code`` given): instead of asking the
    model for a fresh gen-0 combiner, the incumbent is seeded DIRECTLY from an
    already-frozen combiner's ``code`` (+ ``seed_meta`` for its name/strategy/
    description/provenance). Generation 0 does NO LLM call -- it just compiles,
    validates, and evaluates the frozen code on the (typically single) scenario to
    measure its baseline fitness; generations 1..G then run the ordinary improve
    hill-climb on top of it. This specialises a generalist policy to one concrete
    operating point at minimal LLM cost. ``seed_code=None`` (default) is the
    unchanged from-scratch behaviour.
    """
    skill_names = tuple(frozen_skills)
    use_scenarios = scenarios is not None
    reward_mode = bool(reward_spec)
    if use_scenarios and scenario_frames is None:
        scenario_frames = scenario_norm_frames(
            frozen_skills, scenarios, reward_function=reward_function
        )

    def _eval(cand: CombinerCandidate) -> CombinerEval:
        if use_scenarios:
            return evaluate_combiner_scenarios(
                cand.make_combiner(), frozen_skills, scenarios, scenario_frames,
                objective=objective, reward_function=reward_function,
            )
        return evaluate_combiner(
            cand.make_combiner(), frozen_skills, train_prefs, ranges,
            objective=objective,
            regimes=regimes, split=split, num_drivers=num_drivers,
            order_limit=order_limit, seed=seed,
        )

    # --- Generation 0: propose (or warm-start from a frozen combiner). ----- #
    if seed_code is not None:
        best = _seed_candidate(seed_code, seed_meta or {}, skill_names,
                               reward_mode=reward_mode)
        best.evaluation = _eval(best)
        log(
            f"[gen 0] WARM-START {best.name!r} strategy={best.meta['strategy']!r} "
            f"fitness={best.evaluation.fitness:.4g} "
            f"(raw {best.evaluation.raw_fitness:.4g}, fallback {best.evaluation.fallback_rate:.2f}, "
            f"stepTVD {best.evaluation.max_step_tvd:.2f})"
        )
    else:
        best = _propose_with_repair(
            client,
            lambda fb: build_combiner_prompt(
                env_profile, frozen_cards,
                scene_variability=use_scenarios, reward_spec=reward_spec,
                ignore_pref=ignore_pref, probe_event_evolve=probe_event_evolve,
                repair_feedback=fb,
            ),
            gen=0, skill_names=skill_names, reward_mode=reward_mode,
            temperature=temperature, log=log,
            independent_events=independent_events,
            probe_event_evolve=probe_event_evolve,
        )
        best.evaluation = _eval(best)
        log(
            f"[gen 0] {best.name!r} strategy={best.meta['strategy']!r} "
            f"fitness={best.evaluation.fitness:.4g} "
            f"(raw {best.evaluation.raw_fitness:.4g}, fallback {best.evaluation.fallback_rate:.2f}, "
            f"stepTVD {best.evaluation.max_step_tvd:.2f})"
        )

    # --- Generations 1..G: improve. --------------------------------------- #
    for gen in range(1, generations + 1):
        for _ in range(lam):
            try:
                cand = _propose_with_repair(
                    client,
                    lambda fb: build_combiner_prompt(
                        env_profile, frozen_cards,
                        current_code=best.meta["code"],
                        current_fitness=best.evaluation.fitness,
                        scene_variability=use_scenarios,
                        reward_spec=reward_spec,
                        ignore_pref=ignore_pref,
                        repair_feedback=fb,
                    ),
                    gen=gen, skill_names=skill_names, reward_mode=reward_mode,
                    temperature=temperature, log=log,
                    independent_events=independent_events,
                )
            except EvolutionError as e:
                log(f"[gen {gen}] candidate failed: {e}")
                continue
            cand.evaluation = _eval(cand)
            improved = cand.evaluation.fitness > best.evaluation.fitness
            log(
                f"[gen {gen}] {cand.name!r} fitness={cand.evaluation.fitness:.4g} "
                f"(stepTVD {cand.evaluation.max_step_tvd:.2f}) "
                f"({'ACCEPT' if improved else 'reject'})"
            )
            if improved:
                best = cand

    return best


# =========================================================================== #
# Final version (B2): evolve ONE combiner across a DISTRIBUTION of objectives.  #
# =========================================================================== #


def select_survivors(
    pool: Sequence[CombinerCandidate],
    objectives: Sequence[object],
    *,
    mu: int = 4,
    family_beta: float = DEFAULT_FAMILY_BETA,
    log: Callable[[str], None] = lambda _s: None,
) -> List[CombinerCandidate]:
    """The ``mu`` best of ``pool`` by :func:`selection_score`, plus family elites.

    After the plain top-``mu`` cut, every objective family present in ``objectives``
    gets ONE reserved slot: the pool member with the highest ``per_family[family]``
    survives even when its overall selection key is mediocre. Without this, the
    round's mean-plus-weakest-family key still culls a program that is the best
    thing anyone has on ONE hard family (the gate's scarce-fleet completion cells)
    but unremarkable elsewhere -- and that program is exactly the crossover
    material a strong all-rounder needs. The archive is therefore at most
    ``mu + n_families`` and usually smaller (elites are normally already inside
    the top ``mu``).
    """
    def _sel(c: CombinerCandidate) -> float:
        return selection_score(c.evaluation, beta=family_beta) \
            if c.evaluation is not None else float("-inf")

    ranked = sorted(pool, key=_sel, reverse=True)
    keep: List[CombinerCandidate] = list(ranked[:max(1, mu)])
    kept = {id(c) for c in keep}
    for fam in sorted({getattr(o, "family", "?") for o in objectives}):
        scored = [c for c in pool
                  if c.evaluation is not None and fam in c.evaluation.per_family]
        if not scored:
            continue
        champ = max(scored, key=lambda c: c.evaluation.per_family[fam])
        if id(champ) not in kept:
            keep.append(champ)
            kept.add(id(champ))
            log(f"    [elite] {champ.name!r} survives on family {fam} "
                f"({champ.evaluation.per_family[fam]:.2f}) despite selection "
                f"{_sel(champ):.3f}")
    return keep


def _roll_baseline(
    frozen_skills: Dict[str, Skill],
    scs: Sequence[Scenario],
    objs: Sequence[object],
    *,
    workers: int,
    blend_k: int = DEFAULT_BLEND_K,
    log: Callable[[str], None] = print,
) -> List[float]:
    """The round's "no choice was made" reward, one number per (scene, objective).

    Every frozen skill weighted equally for every driver under every objective --
    what :class:`~pref_dispatch.combiner.EqualBlendCombiner` runs, and exactly
    what a fully-broken candidate falls back to. The Phase-2 fitness subtracts
    this, so a program that never runs scores exactly 0 and one that runs but
    chooses badly goes NEGATIVE. Costs one extra full-hour rollout per pair --
    the same order as one more candidate in the pool.

    Parallel when ``workers > 1``, with the same in-process fallback as the
    candidate rollouts (a half-run round is worse than a slow one).
    """
    if workers > 1 and len(scs):
        try:
            return parallel_baseline_rows(frozen_skills, scs, objs,
                                          workers=workers, blend_k=blend_k,
                                          log=log)
        except NotParallelizable as e:
            log(f"    [parallel] baseline not usable this round ({e}); "
                "running in-process")
    return _baseline_rewards(frozen_skills, scs, objs, blend_k=blend_k)


def evolve_combiner_objectives(
    client: LLMClient,
    env_profile: str,
    frozen_skills: Dict[str, Skill],
    frozen_cards: Sequence[Dict],
    scenarios: Sequence[Scenario],
    objectives: Sequence[object],
    *,
    batch_fn: Optional[Callable[[int], "tuple"]] = None,
    seed_code: Optional[str] = None,
    seed_meta: Optional[Dict] = None,
    generations: int = 4,
    mu: int = 4,
    lam: int = 4,
    crossover_rate: float = 0.35,
    fresh_per_round: int = 1,
    rng: Optional[random.Random] = None,
    skill_refs_in_group: bool = False,
    temperature: float = 0.9,
    fallback_penalty: float = 0.0,
    family_beta: float = DEFAULT_FAMILY_BETA,
    workers: int = 1,
    checkpoint_fn: Optional[Callable[[int, "CombinerCandidate"], None]] = None,
    patience: int = 0,
    min_gen: int = 0,
    runoff: bool = False,
    probe_event_evolve: bool = False,
    independent_events: bool = False,
    log: Callable[[str], None] = print,
) -> CombinerCandidate:
    """Evolve ONE objective-reading combiner with a ``(mu+lambda)`` evolution strategy.

    This is the final-version Phase-2 loop (§B2 / v6). Every candidate is scored
    across ``zip(scenarios, objectives)``: each pair injects its OWN reward into the
    env and hands it to the combiner as ``w``.

    **Fitness is GROUP-RELATIVE (GRPO), not ruler-scalarised.** On each
    (scenario, objective) pair a candidate's score is its standardised advantage
    ``(r - mean)/std`` among the programs alive that round; the mean over the batch
    is the raw fitness and only the fallback penalty is subtracted. Objective
    INTERNAL scale therefore cannot matter: two objectives differing by a 2x weight
    give byte-identical advantages (the old per-scenario min/max ruler saturated
    and could not discriminate -- that was the root cause this replaces). v6/v7
    used the percentile rank instead of the advantage; the rank was equally
    scale-free but discarded the margin, so "second by a hair" and "second by a
    mile" selected identically (see :mod:`pref_dispatch.llm.group_fitness`).

    Four v6 changes to how that group is formed:

    * **Scenes rotate every round** (``batch_fn(round)`` returns a fresh
      ``(scenarios, objectives)`` batch), so a policy cannot win by fitting one
      batch of demand windows. Passing ``batch_fn=None`` reuses the fixed batch for
      every round (the pre-v6 behaviour, kept for the offline tests).
    * **Parents are RE-ROLLED with the offspring** on that round's batch, so the
      round's comparison is exact and paired -- every program in it saw the same
      scenes and the same objectives. v5 instead accumulated one row per candidate
      forever and never re-rolled, which made an early candidate's rank an artifact
      of the scenes it happened to be admitted on.
    * **Single-skill references are NOT in the selection group** (``skill_refs_in_group``
      defaults to False). Re-rolling the archive already puts a candidate against
      real competitors on the same objective; paying 8 extra full-hour rollouts per
      pair to also rank it against the frozen skills bought nothing selection did
      not already have. The single skills survive as a separate fixed-batch
      yardstick (:func:`skill_yardstick`) for the learning curve and the paper's
      "beats N of 8 single skills" number.
    * **A fallback is a bug, not a cost.** Any program whose ``skill_scores``
      raised or returned nothing usable for even one driver gets ONE targeted
      repair call carrying the actual cause (``LLMCombiner.first_fallback_reason``,
      e.g. ``KeyError: 'idle_min'``); the fix REPLACES it in the round and is
      re-rolled. If it still falls back it is ELIMINATED -- it cannot survive the
      round no matter how it ranks. ``fallback_penalty`` defaults to 0 and is kept
      only as a legacy knob: since 2026-08-12 a fallback runs the EQUAL BLEND,
      which is the fitness baseline itself, so a program that breaks on every
      driver scores exactly 0.0 without any coefficient. It used to inherit
      ``skill_names[0]`` -- a working single-skill policy -- and the penalty
      existed to charge back that borrowed credit.

    Survivors each round are the ``mu`` best by :func:`selection_score` (with the
    user-required default ``family_beta = 0`` that is EXACTLY the pure GRPO mean
    advantage) PLUS one reserved elite slot per objective family: the best program
    on a family survives even if its overall mean is mediocre, so a genuine
    specialist on a hard family (the gate's completion cells) is not culled before
    crossover can harvest it. Offspring are produced by an LLM MUTATION of one
    random survivor or -- with probability ``crossover_rate``, when at least two
    survive -- an LLM CROSSOVER of two random survivors, which is how a
    specialist's mechanism reaches a strong all-rounder instead of dying with it.

    ``objectives`` are
    :class:`~pref_dispatch.llm.objective_sampler.SampledObjective` s;
    ``scenarios`` is a same-length batch. Without a runoff the returned candidate
    is the best of the FINAL round's group -- advantages from different rounds are
    measured on different scenes and are not comparable, so the last paired
    comparison is the verdict.

    **Adaptive stopping (``patience``, 2026-08-13).** ``patience = K > 0`` stops
    the loop early when the SAME program (byte-identical ``code``) has been the
    round leader for ``K`` consecutive rounds -- the search has converged on one
    answer and further rounds only re-measure it. ``generations`` becomes the hard
    safety cap. ``patience = 0`` (default) keeps the fixed-length behaviour.

    **Minimum rounds (``min_gen``, 2026-08-25).** When ``min_gen > 0``, the
    adaptive stop only fires at/after round ``min_gen``: before that the search
    always runs to ``generations`` even if the leader is stable, so a fast-but-
    stable champion is not accepted while the search has barely explored. The
    stop is ``gen >= min_gen and streak >= patience``; if the champion never
    stabilises by ``generations``, that cap ends the run.

    **Runoff final (``runoff``, 2026-08-13).** Each round's leader is remembered
    (deduplicated by code). With ``runoff = True``, after the loop stops, all
    remembered round-leaders are re-rolled TOGETHER on ONE fresh batch drawn from
    ``batch_fn`` -- a single paired GRPO comparison on scenes none of them was
    selected on -- and the runoff winner is returned. This removes the "last
    round's draw decides the champion" lottery: a leader that won round 3 and a
    leader that won round 8 meet on equal terms. Pure group-relative comparison,
    no hand-set constant. Requires ``batch_fn`` (a fixed batch would re-use seen
    scenes); silently skipped when there is only one distinct leader.

    ``workers > 1`` spreads the round's ``(mu + lambda) x len(batch)`` rollouts over
    that many processes (:mod:`pref_dispatch.llm.parallel`). Every rollout is seeded
    by its own scenario, so the scores do not depend on how the work was split; if a
    candidate or objective cannot be shipped to a worker the round silently falls
    back to the in-process loop. ``workers <= 1`` never touches that path.

    ``checkpoint_fn(generation, leader)`` fires after every round's selection.
    Artifacts are only written when the whole phase RETURNS, so without this an API
    outage in round 9 of 10 loses ten hours of evolved programs; see
    :class:`pref_dispatch.llm.checkpoint.LeaderCheckpoint`, which writes them under
    ``cache/`` rather than into the live ``evolved/`` load path.
    """
    if len(scenarios) != len(objectives):
        raise ValueError("scenarios and objectives must be the same length")
    if mu < 1 or lam < 1:
        raise ValueError("mu and lam must both be >= 1")
    skill_names = tuple(frozen_skills)
    rnd = rng or random.Random(0)

    def _batch(round_idx: int):
        """This round's (scenarios, objectives); the fixed batch when no sampler."""
        if batch_fn is None:
            return list(scenarios), list(objectives)
        scs, objs = batch_fn(round_idx)
        if len(scs) != len(objs):
            raise ValueError(f"round {round_idx}: batch_fn returned "
                             f"{len(scs)} scenarios vs {len(objs)} objectives")
        return list(scs), list(objs)

    def _sel(c: CombinerCandidate) -> float:
        """Family-aware selection key (see :func:`selection_score`)."""
        return selection_score(c.evaluation, beta=family_beta) \
            if c.evaluation is not None else float("-inf")

    def _repair_runtime(cand: CombinerCandidate, gen: int
                        ) -> Optional[CombinerCandidate]:
        """One targeted fix for a program that fell back at RUNTIME.

        Validation-time repair (:func:`_propose_with_repair`) only proves the code
        parses and has the right shape; a program can still raise on the ten-
        thousandth driver because one obs key is missing in a scarce-fleet scene.
        That is a bug, not a cost, so v6 spends exactly one LLM call on it with the
        real cause attached -- and eliminates the program if the fix does not take.
        """
        why = getattr(cand.combiner, "first_fallback_reason", None) or "unknown cause"
        note = (f"RUNTIME FAILURE: your skill_scores raised or returned nothing "
                f"usable on {cand.combiner.fallback_rate:.1%} of driver decisions, "
                f"so those drivers silently fell back to one default skill. First "
                f"cause: {why}. Fix THAT -- read every obs field with .get(key, "
                f"default), never assume a key exists, and always return a finite "
                f"float for every skill name. Keep the strategy identical.")
        try:
            fixed = _propose_with_repair(
                client,
                lambda fb: build_combiner_prompt(
                    env_profile, frozen_cards,
                    current_code=cand.meta["code"],
                    current_fitness=0.0,
                    current_fitness_note=note,
                    scene_variability=True, reward_spec=None,
                    ignore_pref=False, repair_feedback=fb or note,
                ),
                gen=gen, skill_names=skill_names, reward_mode=False,
                temperature=min(temperature, _REPAIR_MIN_TEMPERATURE + 0.2),
                log=log,
            )
        except EvolutionError as e:
            log(f"    [runtime-repair] {cand.name!r} unrecoverable: {e}")
            return None
        fixed.meta["operator"] = f"runtime-repair({cand.name})"
        fixed.meta["parents"] = [cand.name]
        return fixed

    def _roll_rows(pool: Sequence[CombinerCandidate], scs, objs):
        """Roll every candidate on every pair; return ``(rows, blindness)``.

        On one core this is the plain loop. With ``workers > 1`` the same rollouts
        run in worker processes and the telemetry each one measured (fallback
        counts, first cause, the fleet-mix probe) is copied back onto the parent's
        combiner objects, so everything downstream reads exactly the fields it
        read before. Anything that cannot be shipped falls back to the loop rather
        than half-running the round.
        """
        for c in pool:
            c.combiner = c.make_combiner()
        if workers > 1 and pool:
            try:
                recs = parallel_pair_rewards(
                    pool, frozen_skills, scs, objs, workers=workers, log=log)
            except NotParallelizable as e:
                log(f"    [parallel] not usable this round ({e}); running in-process")
            else:
                blind = []
                for c, rec in zip(pool, recs):
                    c.combiner.n_calls = rec["n_calls"]
                    c.combiner.n_fallbacks = rec["n_fallbacks"]
                    c.combiner.first_fallback_reason = rec["reason"]
                    blind.append(blindness_from_dists(rec["picks"] or []))
                return [rec["rewards"] for rec in recs], blind
        rows = [_roll_pair_rewards(c.combiner, frozen_skills, scs, objs)
                for c in pool]
        return rows, None

    def _score_pool(pool: List[CombinerCandidate], scs, objs, gen: int) -> set:
        """Roll EVERY member of ``pool`` on this round's batch and rank the group.

        Parents and offspring alike -- that is what makes the round's comparison a
        paired one. Each candidate gets a fresh :class:`LLMCombiner` so its
        telemetry (fallback rate, captured driver-obs sample) belongs to THIS
        round's rollouts and not to an older batch.

        Any program that fell back gets ONE :func:`_repair_runtime` attempt and is
        REPLACED in ``pool`` by the fix; whatever still falls back afterwards is
        returned in the eliminated set and cannot survive the round. Returns the
        set of eliminated ``id()``s.
        """
        rows, blind = _roll_rows(pool, scs, objs)
        broken = set()
        # --- BEHAVIOURAL CLONE KILL (2026-08-10) --------------------------- #
        # A candidate's row is its episode reward on every (scene, objective) pair
        # of the round: its behavioural fingerprint. In the v6 run gen 2 carried
        # THREE distinct source programs with byte-identical rows -- three of the
        # round's slots spent re-measuring one behaviour, and a ranking group
        # padded with copies of yourself (which is what pinned whole families at
        # exactly 0.500). Keep the FIRST occurrence in pool order -- the archive
        # comes first, so a surviving parent always outranks a child that merely
        # reproduced it -- and eliminate the rest.
        seen_rows: Dict[tuple, CombinerCandidate] = {}
        for c, row in zip(pool, rows):
            key = tuple(row)
            first = seen_rows.get(key)
            if first is None:
                seen_rows[key] = c
                continue
            broken.add(id(c))
            log(f"    [clone] {c.name!r} is behaviourally IDENTICAL to "
                f"{first.name!r} on all {len(row)} pairs -- ELIMINATED this round")
        for i, c in enumerate(list(pool)):
            if c.combiner.fallback_rate <= 0.0:
                continue
            log(f"    [runtime] {c.name!r} fell back on "
                f"{c.combiner.fallback_rate:.1%} of decisions "
                f"({c.combiner.first_fallback_reason}) -- one repair attempt")
            fixed = _repair_runtime(c, gen)
            if fixed is not None:
                one_row, one_blind = _roll_rows([fixed], scs, objs)
                rows[i] = one_row[0]
                if blind is not None and one_blind is not None:
                    blind[i] = one_blind[0]
                pool[i] = fixed
                c = fixed
            if c.combiner.fallback_rate > 0.0:
                broken.add(id(c))
                log(f"    [runtime] {c.name!r} still falls back "
                    f"({c.combiner.fallback_rate:.1%}) -- ELIMINATED this round")
        for row in rows:
            assert len(row) == len(scs), (
                f"rollout row {len(row)} does not match the round's grid {len(scs)}")
        refs = (_skill_reference_rewards(frozen_skills, scs, objs)
                if skill_refs_in_group else [[] for _ in scs])
        base = _roll_baseline(frozen_skills, scs, objs, workers=workers, log=log)
        evals = _group_evals(rows, refs, objs,
                             baseline=base,
                             combiners=[c.combiner for c in pool],
                             fallback_penalty=fallback_penalty,
                             blindness=blind)
        for c, ev in zip(pool, evals):
            c.evaluation = ev
        return broken

    def _survivors(pool: Sequence[CombinerCandidate], objs: Sequence[object],
                   broken: set) -> List[CombinerCandidate]:
        """The ``mu`` best by selection key, plus one elite slot per family.

        Eliminated (still-falling-back) programs are excluded outright -- unless
        that would empty the archive, in which case the round has nothing better
        and they stay so the run can continue."""
        alive = [c for c in pool if id(c) not in broken] or list(pool)
        return select_survivors(alive, objs, mu=mu, family_beta=family_beta, log=log)

    def _fitness_note(cand: CombinerCandidate) -> str:
        ev = cand.evaluation
        fam = " ".join(f"{k} {v:.2f}" for k, v in sorted(ev.per_family.items()))
        mn = min(ev.per_family.values(), default=0.0)
        worst = min(ev.per_family, key=ev.per_family.get) if ev.per_family else "?"
        # Degeneracy read-out. Under the delta fitness a 0.00 is no longer
        # ambiguous about the BASELINE (it always means "worth exactly as much as
        # not choosing"), but it is still ambiguous about the ROUND: a whole
        # column that never diverged is a search that has stalled, and the v6 run
        # was that case for five straight generations. Name the dead families and
        # give the absolute bar (how many single skills cleared it), so the model
        # gets an actionable fact instead of a score that reads as "average".
        dead = [f for f, t in sorted(ev.family_tie_rate.items()) if t >= 0.5]
        deg = ""
        if dead:
            bits = []
            for f in dead:
                above = ev.family_skills_above.get(f, 0.0)
                bits.append(f"{f} (identical to {ev.family_tie_rate[f]:.0%} of the "
                            f"round; {above:.1f} frozen single skills BEAT you there)")
            deg = (" DEAD FAMILIES -- your fleet produced the SAME episode reward as "
                   "the rest of the round on: " + "; ".join(bits) +
                   ". That means no program in this round -- including yours -- did "
                   "anything different when the objective changed shape. Reading `w` "
                   "and then acting identically is the failure this run is trying to "
                   "fix. Make the fleet's skill choice VISIBLY different under "
                   "those objective shapes; even an imperfect switch scores above "
                   "the tied block, and a single skill already beats you there.")
        return (
            f"SELECTION {_sel(cand):.3f} = mean gain-over-not-choosing "
            f"{ev.raw_fitness:.2f} (pure GRPO advantage; no family term -- "
            f"beta is 0). per-family: {fam or 'n/a'}; YOUR WEAKEST FAMILY: {worst} "
            f"{mn:.2f} -- a specialist survives via the reserved per-family slot, "
            f"not via any weight on the weakest family. Your score on an objective is "
            f"(YOUR episode reward - the reward of the EQUAL BLEND, every frozen "
            f"skill weighted the same for every driver, on the SAME scene and "
            f"seed) / how much this round's programs disagree about that same "
            f"difference. So reward SCALES do NOT matter, and the sign is "
            f"absolute: 0.00 means your choosing was worth exactly as much as not "
            f"choosing at all, NEGATIVE means picking skills per driver ACTIVELY "
            f"LOST money against a flat blend, and +1 means your gain is one whole "
            f"spread above how much the field's gains vary. Beating the other "
            f"candidates is NOT the target -- beating 'no choice was made' is. "
            f"broke on {ev.fallback_rate:.2f} of decisions (a raise falls back to "
            f"that same equal blend, so crashing scores 0, never some single "
            f"skill's result), deferred on {ev.defer_rate:.2f}. objective "
            f"blindness {ev.objective_blindness:.2f} (0 = the fleet's skill mix "
            f"visibly moves between objectives, 1 = it never moves)." + deg
        )

    def _fmt_rank(ev: CombinerEval) -> str:
        fam = " ".join(f"{k} {v:.2f}" for k, v in sorted(ev.per_family.items()))
        # Degeneracy in the LOG too: a run whose families are all tied looks
        # healthy in the old format (every number near 0.00) and is not.
        tied = ",".join(f for f, t in sorted(ev.family_tie_rate.items()) if t >= 0.5)
        # Report the numbers selection ACTUALLY uses: the fitness after the
        # fallback penalty and the selection key itself, not just the raw
        # advantage -- reading a run's accept/reject decisions from the log is
        # impossible otherwise.
        return (f"adv {ev.raw_fitness:+.2f} -> fitness {ev.fitness:.3f} "
                f"(selection {selection_score(ev, beta=family_beta):.3f}; "
                f"per-family: {fam or 'n/a'}, "
                f"fallback {ev.fallback_rate:.2f}, "
                f"blindness {ev.objective_blindness:.2f}"
                + (f", TIED-FAMILIES {tied}" if tied else "") + ")")

    def _parent_card(c: CombinerCandidate) -> Dict:
        return {"name": c.name, "strategy": c.meta.get("strategy", ""),
                "code": c.meta["code"], "fitness_note": _fitness_note(c)}

    def _offspring(gen: int, survivors: Sequence[CombinerCandidate]
                   ) -> Optional[CombinerCandidate]:
        """One LLM child: crossover of two random survivors, else mutation of one."""
        do_cross = len(survivors) >= 2 and rnd.random() < crossover_rate
        if do_cross:
            pa, pb = rnd.sample(list(survivors), 2)
            kind = f"crossover({pa.name} x {pb.name})"
            cards = [_parent_card(pa), _parent_card(pb)]
            build = lambda fb: build_combiner_prompt(  # noqa: E731
                env_profile, frozen_cards, parents=cards,
                scene_variability=True, reward_spec=None,
                ignore_pref=False, repair_feedback=fb,
                probe_event_evolve=probe_event_evolve,
            )
        else:
            pa = rnd.choice(list(survivors))
            kind = f"mutation({pa.name})"
            build = lambda fb: build_combiner_prompt(  # noqa: E731
                env_profile, frozen_cards,
                current_code=pa.meta["code"],
                current_fitness=pa.evaluation.fitness,
                current_fitness_note=_fitness_note(pa),
                scene_variability=True, reward_spec=None,
                ignore_pref=False, repair_feedback=fb,
                probe_event_evolve=probe_event_evolve,
            )
        try:
            cand = _propose_with_repair(
                client, build, gen=gen, skill_names=skill_names,
                reward_mode=False, temperature=temperature, log=log,
                probe_event_evolve=probe_event_evolve,
            )
        except EvolutionError as e:
            log(f"[gen {gen}] {kind} failed: {e}")
            return None
        cand.meta["operator"] = kind
        cand.meta["parents"] = ([pa.name, pb.name] if do_cross else [pa.name])
        return cand

    def _fresh_child(gen: int) -> Optional[CombinerCandidate]:
        """One PARENTLESS program: proposed from the task alone, no parent shown.

        Every mutation/crossover child descends from a gen-0 program, so after a
        few rounds the whole archive is one template wearing different thresholds
        -- in the v6 run every survivor traced back to a single
        ``objective_shape_*`` ancestor and routed to the same skill under
        completion-shaped ``w``. This slot is the only way a genuinely different
        mechanism can enter after round 0. It competes on the same footing: if it
        is worse than the archive it simply does not survive, so the cost is at
        most one rollout row per round.
        """
        try:
            cand = _propose_with_repair(
                client,
                lambda fb: build_combiner_prompt(
                    env_profile, frozen_cards,
                    scene_variability=True, reward_spec=None,
                    ignore_pref=False, repair_feedback=fb,
                    probe_event_evolve=probe_event_evolve,
                ),
                gen=gen, skill_names=skill_names, reward_mode=False,
                temperature=temperature, log=log,
                probe_event_evolve=probe_event_evolve,
            )
        except EvolutionError as e:
            log(f"[gen {gen}] fresh (parentless) proposal failed: {e}")
            return None
        cand.meta["operator"] = "fresh(parentless)"
        cand.meta["parents"] = []
        return cand

    log(f"[B2e] (mu={mu}+lambda={lam}) group-relative evolution, "
        f"{generations} round(s) x {len(objectives)} (scene, objective) pairs; "
        f"scenes {'ROTATE per round' if batch_fn else 'FIXED across rounds'}; "
        f"single-skill refs {'IN' if skill_refs_in_group else 'OUT OF'} the group; "
        f"crossover rate {crossover_rate}")

    # --- Round 0: fill the archive. ---------------------------------------- #
    # reward_spec is None: the combiner reads w at runtime, it is not built for one
    # reward. reward_mode stays False (no reward_understanding CoT gate here).
    scs, objs = _batch(0)
    pool: List[CombinerCandidate] = []
    if seed_code is not None:
        pool.append(_seed_candidate(seed_code, seed_meta or {}, skill_names,
                                    reward_mode=False))
    while len(pool) < mu:
        try:
            pool.append(_propose_with_repair(
                client,
                lambda fb: build_combiner_prompt(
                    env_profile, frozen_cards,
                    scene_variability=True, reward_spec=None,
                    ignore_pref=False, repair_feedback=fb,
                    probe_event_evolve=probe_event_evolve,
                ),
                gen=0, skill_names=skill_names, reward_mode=False,
                temperature=temperature, log=log,
                probe_event_evolve=probe_event_evolve,
            ))
        except EvolutionError as e:
            log(f"[gen 0] proposal failed: {e}")
            if not pool:
                raise
            break
    broken = _score_pool(pool, scs, objs, 0)
    for c in sorted(pool, key=_sel, reverse=True):
        log(f"[gen 0] {c.name!r} strategy={c.meta['strategy']!r} "
            f"{_fmt_rank(c.evaluation)}"
            f"{' [ELIMINATED: falls back]' if id(c) in broken else ''}")
    archive = _survivors(pool, objs, broken)
    if checkpoint_fn is not None:
        checkpoint_fn(0, max(archive, key=_sel))

    # Round leaders for the runoff final, deduplicated by CODE (the same program
    # leading five rounds enters once), plus the streak counter for patience.
    round_leaders: List[CombinerCandidate] = []
    leader_codes: set = set()

    def _remember_leader(champ: CombinerCandidate) -> None:
        code = str(champ.meta.get("code", ""))
        if code and code not in leader_codes:
            leader_codes.add(code)
            round_leaders.append(champ)

    _remember_leader(max(archive, key=_sel))
    streak_code: Optional[str] = str(max(archive, key=_sel).meta.get("code", ""))
    streak = 1

    # --- Rounds 1..G: mutate/cross, re-roll EVERYTHING, reselect. ---------- #
    for gen in range(1, generations + 1):
        scs, objs = _batch(gen)
        # ``fresh_per_round`` of the lam slots are parentless injections; the rest
        # are mutation/crossover of survivors.
        n_fresh = max(0, min(int(fresh_per_round), lam))
        raw_children = ([_fresh_child(gen) for _ in range(n_fresh)]
                        + [_offspring(gen, archive) for _ in range(lam - n_fresh)])
        children = [c for c in raw_children if c is not None]
        if not children:
            log(f"[gen {gen}] no valid offspring; archive re-rolled anyway")
        pool = list(archive) + children
        alive = {id(c) for c in archive}
        broken = _score_pool(pool, scs, objs, gen)
        for c in sorted(pool, key=_sel, reverse=True):
            tag = "parent" if id(c) in alive else c.meta.get("operator", "child")
            log(f"[gen {gen}] {c.name!r} [{tag}] {_fmt_rank(c.evaluation)}"
                f"{' [ELIMINATED: falls back]' if id(c) in broken else ''}")
        archive = _survivors(pool, objs, broken)
        champ = max(archive, key=_sel)
        log(f"[gen {gen}] archive {[c.name for c in archive]}; "
            f"leader {champ.name!r} selection {_sel(champ):.3f}")
        if checkpoint_fn is not None:
            checkpoint_fn(gen, champ)
        _remember_leader(champ)

        # Adaptive stop: only AFTER ``min_gen`` rounds may the same program
        # leading ``patience`` consecutive rounds stop the search. Before the
        # minimum, the search always continues to the hard ``generations`` cap.
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
        scs, objs = _batch(runoff_idx)
        log(f"[runoff] {len(round_leaders)} distinct round leader(s) re-rolled "
            f"together on a FRESH batch of {len(scs)} (scene, objective) pair(s): "
            f"{[c.name for c in round_leaders]}")
        pool = list(round_leaders)
        broken = _score_pool(pool, scs, objs, runoff_idx)
        for c in sorted(pool, key=_sel, reverse=True):
            log(f"[runoff] {c.name!r} {_fmt_rank(c.evaluation)}"
                f"{' [ELIMINATED: falls back]' if id(c) in broken else ''}")
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


def skill_yardstick(
    champion: CombinerCandidate,
    frozen_skills: Dict[str, Skill],
    scenarios: Sequence[Scenario],
    objectives: Sequence[object],
    *,
    fallback_penalty: float = 0.0,
    workers: int = 1,
    log: Callable[[str], None] = print,
) -> Dict[str, object]:
    """Score ``champion`` against every frozen skill rolled alone, on a FIXED batch.

    The single-skill rollouts left the selection group in v6 (re-rolling the archive
    each round already puts a candidate against real competitors on the same
    objective, and the references cost 8 extra full-hour rollouts per pair). They
    are still the only interpretable external scale the paper has, so they survive
    HERE: one fixed batch, rolled once, giving the headline "the evolved combiner
    beats N of the 8 frozen skills" number and a comparable learning-curve point
    across rounds (the training batches rotate, so training scores are not
    comparable between rounds -- this one is).

    Returns ``{"rank", "beaten", "n_skills", "per_family", "skill_rewards",
    "champion_rewards"}``; ``rank`` is the champion's mean standardised advantage
    against the skills alone (the key name predates the 2026-08-10 switch from
    percentile to advantage and is kept so old result files stay readable).
    ``workers > 1`` spreads its ``(n_skills + 1) x len(batch)`` rollouts over
    processes, same rule as the training loop.
    """
    comb = champion.make_combiner()
    blind = None
    if workers > 1:
        try:
            refs = parallel_skill_rows(frozen_skills, scenarios, objectives,
                                       workers=workers, log=log)
            recs = parallel_pair_rewards([champion], frozen_skills, scenarios,
                                         objectives, workers=workers, log=log)
        except NotParallelizable as e:
            log(f"    [parallel] yardstick not usable ({e}); running in-process")
            workers = 1
        else:
            rec = recs[0]
            row = rec["rewards"]
            comb.n_calls = rec["n_calls"]
            comb.n_fallbacks = rec["n_fallbacks"]
            comb.first_fallback_reason = rec["reason"]
            blind = [blindness_from_dists(rec["picks"] or [])]
    if workers <= 1:
        refs = _skill_reference_rewards(frozen_skills, scenarios, objectives)
        row = _roll_pair_rewards(comb, frozen_skills, scenarios, objectives)
    base = _roll_baseline(frozen_skills, scenarios, objectives,
                          workers=workers, log=log)
    evals = _group_evals([row], refs, objectives, baseline=base,
                         combiners=[comb],
                         fallback_penalty=fallback_penalty, blindness=blind)
    ev = evals[0]
    # "Beaten" is per-skill over the whole batch: a skill counts as beaten when the
    # champion's mean reward across the batch exceeds that skill's.
    names = list(frozen_skills)
    skill_means = {n: sum(refs[p][k] for p in range(len(refs))) / max(1, len(refs))
                   for k, n in enumerate(names)}
    champ_mean = sum(row) / max(1, len(row))
    beaten = [n for n, v in skill_means.items() if champ_mean > v]
    log(f"[yardstick] {champion.name!r} mean advantage {ev.raw_fitness:+.2f} vs the "
        f"{len(names)} frozen skills on {len(scenarios)} fixed pairs; beats "
        f"{len(beaten)}/{len(names)} ({', '.join(beaten) or 'none'})")
    return {
        "rank": ev.raw_fitness,
        "beaten": beaten,
        "n_skills": len(names),
        "per_family": dict(ev.per_family),
        "skill_rewards": skill_means,
        "champion_rewards": champ_mean,
    }



def _unique_frozen_name(name: str, out_dir: str) -> str:
    """Return ``name``, or ``name_r2``/``_r3``... if that artifact already exists.

    The model picks the combiner name, and it frequently reuses a good one: a v4
    run proposed ``objective_shape_dispatcher_v3``, the exact name of the frozen
    v2 champion. Writing ``{name}.py`` unconditionally would have destroyed the
    earlier artifact -- and ``evolved/`` is untracked, so git could not recover
    it. Frozen artifacts are experiment records; never overwrite one silently.
    """
    if not os.path.exists(os.path.join(out_dir, f"{name}.py")):
        return name
    n = 2
    while os.path.exists(os.path.join(out_dir, f"{name}_r{n}.py")):
        n += 1
    return f"{name}_r{n}"


def freeze_combiner(
    cand: CombinerCandidate,
    out_dir: str = FrozenDir,
    *,
    reward_snapshot: Optional[Dict] = None,
    reward_provenance: Optional[Dict] = None,
    ignore_pref: bool = False,
) -> str:
    """Write the winning combiner to disk as a runnable module + meta.json.

    ``reward_snapshot`` (reward-conditioned arm): a dict of the reward function's
    live coefficients the combiner was composed FOR, recorded into the meta so a
    reader can verify which reward_func this strategy targets.

    ``reward_provenance`` (§Phase-2 single-reward arm): the full authored-reward
    record -- ``{reward_name, code, reward_understanding, objective, description,
    authored, preference_spec}`` from :class:`~pref_dispatch.llm.evolve_reward.AuthoredReward`.
    Embedded so the frozen strategy carries the EXACT reward it maximises (authored
    body or given-instance snapshot), not just coefficients.

    ``ignore_pref``: records that this combiner ignores its ``pref`` argument by
    construction (one reward -> one strategy, no runtime dial), so a loader / report
    knows not to sweep a preference axis on it.
    """
    os.makedirs(out_dir, exist_ok=True)
    name = _unique_frozen_name(cand.name, out_dir)
    py_path = os.path.join(out_dir, f"{name}.py")
    meta_path = os.path.join(out_dir, f"{name}.meta.json")

    ru = cand.meta.get("reward_understanding")
    reward_block = (
        f"Reward understanding (LLM CoT): {ru}\n\n" if ru else ""
    )
    # For the single-reward arm, embed the target reward's own body/objective so the
    # frozen module is self-describing about WHAT it maximises.
    target_block = ""
    if reward_provenance is not None:
        rp_obj = reward_provenance.get("objective", "")
        rp_code = reward_provenance.get("code")
        target_block = (
            f"Target reward: {reward_provenance.get('reward_name', '?')} -- {rp_obj}\n\n"
        )
        if reward_provenance.get("reward_understanding"):
            target_block += (
                f"Reward understanding (author CoT): "
                f"{reward_provenance['reward_understanding']}\n\n"
            )
        if rp_code:
            target_block += f"Target reward source:\n{rp_code}\n\n"
    pref_note = (
        "This combiner IGNORES its `pref` argument by construction: one fixed reward\n"
        "-> one strategy, no runtime preference dial.\n"
        if ignore_pref else
        "Paradigm B: reads the platform\npreference and adapts with ZERO retraining, "
        "at ~zero online LLM cost."
    )
    header = (
        f'"""Frozen evolved upper combiner: {name}\n\n'
        f"Strategy: {cand.meta['strategy']}\n\n"
        f"{cand.meta['description']}\n\n"
        f"{reward_block}"
        f"{target_block}"
        f"Selects over frozen skills: {list(cand.skill_names)}.\n"
        f"Generated in gen {cand.meta['gen']}. "
        + pref_note.rstrip() + '"""\n\n'
        "import math\n"
        "import numpy as np\n\n\n"
    )
    with open(py_path, "w", encoding="utf-8") as f:
        f.write(header + cand.meta["code"].rstrip() + "\n")

    meta = dict(cand.meta)
    # Keep the recorded name identical to the file name: if the artifact was
    # renamed to avoid clobbering an earlier one, ``--combiner-name`` must find
    # it under the name it was actually written as.
    meta["combiner_name"] = name
    if name != cand.name:
        meta["proposed_name"] = cand.name
    meta["skill_names"] = list(cand.skill_names)
    meta["ignore_pref"] = bool(ignore_pref)
    # TRAINING OPERATING POINT (2026-08-10). A combiner is only reproducible
    # against the matcher settings it was evolved under: ``top_k`` decides how
    # many orders each driver ever sees, ``blend_k`` how many skills one decision
    # may mix. Until this date neither was recorded, and evaluation silently ran
    # at top_k=20 while training ran at 60 -- worth about one gate cell. Recording
    # them makes the mismatch detectable instead of invisible; the loader compares
    # and warns.
    meta["train_top_k"] = int(DEFAULT_TOP_K)
    meta["train_blend_k"] = int(DEFAULT_BLEND_K)
    if reward_snapshot is not None:
        meta["reward_snapshot"] = reward_snapshot
    if reward_provenance is not None:
        meta["reward_provenance"] = reward_provenance
    if cand.evaluation is not None:
        meta["fitness"] = cand.evaluation.fitness
        meta["raw_fitness"] = cand.evaluation.raw_fitness
        meta["fallback_rate"] = cand.evaluation.fallback_rate
        meta["max_step_tvd"] = cand.evaluation.max_step_tvd
        meta["smoothness_penalty"] = cand.evaluation.smoothness_penalty
        meta["objective_blindness"] = cand.evaluation.objective_blindness
        meta["per_regime"] = cand.evaluation.per_regime
        meta["per_family"] = cand.evaluation.per_family
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return py_path
