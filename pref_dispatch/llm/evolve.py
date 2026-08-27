"""Phase-1 skill evolution loop (§4): propose -> improve -> freeze.

The user's design: the agent first authors a reward function (``fitness``), then
we train the best skill under that fixed reward, then move to the next skill.
This module implements exactly one skill's evolution run:

1. **Generation 0 (propose).** Ask the model for a skill: name, objective, a
   self-authored ``fitness(metrics)``, the ``score`` code, and natural-language
   explanations of all of it. Compile + sandbox-validate both functions. The
   fitness chosen here becomes the FIXED yardstick for the rest of the run.
2. **Generations 1..G (improve).** Ask the model to rewrite only the scoring
   logic to raise that fixed fitness, showing it the current best code and its
   measured score. Keep the candidate only if it validates AND scores higher
   (a (1+lambda)-style hill-climb; ``lam`` candidates per generation).
3. **Freeze.** Write the winning skill to ``pref_dispatch/evolved/skills/`` as a
   runnable ``<name>.py`` plus a ``<name>.meta.json`` recording the objective,
   fitness, explanations, and provenance -- the frozen paradigm-B product.

Every model reply flows through the same robust path: extract JSON -> require the
natural-language explanation fields (interpretability gate) -> sandbox. A reply
that fails compile/validate is fed back as ``repair_feedback`` for one retry.

The loop is model-agnostic: it takes any :class:`~pref_dispatch.llm.client.LLMClient`
(the real API client, or a fake one for offline end-to-end testing).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from pref_dispatch.llm.client import LLMClient
from pref_dispatch.llm.extract import extract_json, require_explanation
from pref_dispatch.llm.fitness_eval import (
    DEFAULT_REGIMES,
    EVAL_NUM_DRIVERS,
    EVAL_ORDER_LIMIT,
    ScenarioEval,
    SkillEval,
    evaluate_skill,
    evaluate_skill_random_scenarios,
    reference_metrics_for,
    reference_metrics_for_scenarios,
)
from pref_dispatch.llm.prompts.skill_evolve import (
    REQUIRED_EXPLANATION_FIELDS,
    REQUIRED_FIELDS,
    REQUIRED_MECHANISM_FIELDS,
    REQUIRED_SELF_CHECK_FIELDS,
    build_skill_improve_prompt,
    build_skill_prompt,
    detect_axis,
)
from pref_dispatch.llm.repair import (
    client_reply_header,
    dump_unparseable,
    repair_temperature,
)
from pref_dispatch.llm.sandbox import (
    CompiledSkill,
    SandboxError,
    compile_fitness,
    compile_skill,
    fitness_probes,
    normalise_terms,
    validate_fitness,
    validate_skill,
)
from pref_dispatch.skills import Skill

FrozenDir = os.path.join("pref_dispatch", "evolved", "skills")


# --------------------------------------------------------------------------- #
# Evaluation accessors: a Candidate may carry either a v1 SkillEval (fixed
# 800-car / 3-regime point) or a v2 ScenarioEval (random domain-randomized
# scenarios). These three helpers read the fields the loop needs from EITHER,
# so the (1+lambda) hill-climb and the QD signature stay eval-mode agnostic.
# --------------------------------------------------------------------------- #
def eval_select_key(ev) -> float:
    """Scale-free SELECTION key (higher is better).

    Group eval (the final-version Phase-1 loop): the invalid-penalised mean
    standardised advantage. ScenarioEval: the aggregate rescaled score (already a
    Δ-vs-reference under ``rescale="reference"``). SkillEval: the delta vs the
    reference seed when present, else the absolute mean fitness.

    The group branch is duck-typed on purpose -- ``SkillGroupEval`` lives in
    :mod:`pref_dispatch.llm.evolve_skill_group`, which imports this module, so
    naming the class here would be an import cycle.
    """
    if ev is None:
        return float("-inf")
    if hasattr(ev, "per_scenario_adv"):
        return ev.fitness
    if isinstance(ev, ScenarioEval):
        return ev.score
    return ev.delta_vs_reference if ev.delta_vs_reference is not None else ev.fitness_mean


def eval_report_value(ev) -> float:
    """The number logged and compared for the hill-climb ACCEPT/reject decision.

    For scenarios this is the aggregate rescaled score; for the fixed point it is
    the absolute mean fitness (the v1 improvement yardstick); for a group eval it is
    the same advantage the selection key uses.
    """
    if ev is None:
        return float("-inf")
    if hasattr(ev, "per_scenario_adv"):
        return ev.fitness
    if isinstance(ev, ScenarioEval):
        return ev.score
    return ev.fitness_mean


def eval_metrics_list(ev) -> List[Dict[str, float]]:
    """Per-unit episode-metrics dicts (one per regime or per scenario).

    This is what the QD behavioural signature averages over -- so a v2 skill's
    signature is measured across the SAME randomized scenarios it evolved on. A
    group eval carries ``None`` for any scene the skill raised on; those are dropped
    rather than zero-filled, so a crash cannot masquerade as a behaviour.
    """
    if ev is None:
        return []
    if hasattr(ev, "per_scenario_adv"):
        return [m for m in ev.per_scenario_metrics if m]
    if isinstance(ev, ScenarioEval):
        return list(ev.per_scenario_metrics)
    return list(ev.per_regime_metrics.values())


class EvolutionError(RuntimeError):
    """Raised when a generation cannot produce any valid candidate."""


@dataclass
class Candidate:
    """A validated skill plus its provenance and measured fitness."""

    meta: Dict  # skill_name, objective, description, fitness_*, gen, code
    skill: CompiledSkill
    fitness_fn: Callable[[Dict], float]
    # Either a v1 SkillEval (fixed point) or a v2 ScenarioEval (random scenarios).
    evaluation: Optional[object] = None

    @property
    def name(self) -> str:
        return self.meta["skill_name"]

    @property
    def score_value(self) -> float:
        """Selection key: scale-free delta vs the reference (SkillEval) or the
        aggregate rescaled scenario score (ScenarioEval)."""
        return eval_select_key(self.evaluation)


def _check_fields(obj: Dict, *, require_mechanism: bool = False,
                  require_self_check: bool = False) -> None:
    """Require every contract field, and non-empty NL explanation fields.

    ``require_mechanism`` additionally demands ``mechanism`` / ``differs_from`` --
    the group-evolution paths select for behavioural diversity, so a program that
    will not say which decision rule it uses cannot be judged on it. Off by default
    so the legacy hill-climb and the Phase-3 warm start (neither of which selects on
    mechanism) keep working on artifacts frozen before the field existed.
    ``require_self_check`` additionally demands ``objective_self_check`` -- the
    generation-0 authoring path, where the model writes a NEW objective and must
    state which objective axis it covers and that the axis is not already in the
    repository. Off for later variants, whose objective is FIXED.
    """
    required = list(REQUIRED_FIELDS)
    explain = list(REQUIRED_EXPLANATION_FIELDS)
    if require_mechanism:
        required += list(REQUIRED_MECHANISM_FIELDS)
        explain += list(REQUIRED_MECHANISM_FIELDS)
    if require_self_check:
        required += list(REQUIRED_SELF_CHECK_FIELDS)
        explain += list(REQUIRED_SELF_CHECK_FIELDS)
    missing = [f for f in required if not str(obj.get(f, "")).strip()]
    if missing:
        raise SandboxError(f"response missing required field(s): {missing}")
    require_explanation(obj, tuple(explain))
    if require_self_check:
        # Content-level, not just non-empty: the objective must NAME a known axis.
        # The model's own `objective_self_check` prose is not trusted -- we parse
        # the objective/description/mechanism text itself. A generalist reword that
        # maps to no canonical axis is NOT a niche, so it is rejected and the
        # repair loop asks for one. (Whether the axis is saturated in the
        # repository is a separate, behavioural question answered by cosine dedup
        # in qd_basis, since two skills on one axis with different decision rules
        # are both legitimate.)
        text_axes = set()
        for field in ("objective", "description", "mechanism"):
            text_axes.update(detect_axis(str(obj.get(field, ""))))
        if not text_axes:
            raise SandboxError(
                "objective self-check failed: the proposed objective/description "
                "names no known objective axis (revenue, service, throughput, "
                "detour, capacity, empty/idle, fairness, option value). A "
                "reworded generalist is not a niche -- specialise on ONE of these "
                "axes or you are duplicating existing coverage."
            )


def _build_candidate(
    obj: Dict,
    gen: int,
    fixed_fitness_fn=None,
    *,
    fixed_fitness_code: Optional[str] = None,
    fixed_objective: Optional[str] = None,
    require_mechanism: bool = False,
    require_self_check: bool = False,
) -> Candidate:
    """Compile + validate a model response object into a :class:`Candidate`.

    When ``fixed_fitness_fn`` is given (every generation after the objective is
    chosen), the response's fitness is ignored and this fixed one is used instead.
    ``fixed_fitness_code`` / ``fixed_objective`` are then also written into the
    candidate's meta, replacing whatever the model echoed: the fitness the skill was
    actually SCORED by and the text recorded next to it must be the same thing, or a
    frozen artifact documents a yardstick that never graded it. They double as the
    fill-in when the model drops the echo entirely, which would otherwise cost a
    repair call for a field the loop already knows.

    Raises SandboxError on any compile/validation failure (caller turns it into
    repair feedback).
    """
    if fixed_fitness_code and not str(obj.get("fitness_code", "")).strip():
        obj = dict(obj, fitness_code=fixed_fitness_code)
    if fixed_objective and not str(obj.get("objective", "")).strip():
        obj = dict(obj, objective=fixed_objective)
    _check_fields(obj, require_mechanism=require_mechanism,
                  require_self_check=require_self_check)
    name = str(obj["skill_name"]).strip()

    if fixed_fitness_fn is None:
        fitness_fn = compile_fitness(obj["fitness_code"])
        ok, why = validate_fitness(fitness_fn)
        if not ok:
            raise SandboxError(f"fitness invalid: {why}")
        # Normalise the self-authored fitness's term weights to sum to ONE, so a
        # wildly-scaled term is not read as dominant (2026-08-13). Uses the same
        # probe dicts validate_fitness already ran.
        fitness_fn, _norm_scale = normalise_terms(
            fitness_fn, list(fitness_probes().values()), domain="fitness")
    else:
        fitness_fn = fixed_fitness_fn

    skill = compile_skill(obj["code"], name=name)
    ok, why = validate_skill(skill)
    if not ok:
        raise SandboxError(f"skill invalid: {why}")

    meta = {
        "skill_name": name,
        "objective": (fixed_objective or obj["objective"]).strip(),
        "description": obj["description"].strip(),
        "fitness_code": fixed_fitness_code or obj["fitness_code"],
        "fitness_rationale": obj["fitness_rationale"].strip(),
        "code": obj["code"],
        "gen": gen,
    }
    for field_name in REQUIRED_MECHANISM_FIELDS:
        value = str(obj.get(field_name, "")).strip()
        if value:
            meta[field_name] = value
    for field_name in REQUIRED_SELF_CHECK_FIELDS:
        value = str(obj.get(field_name, "")).strip()
        if value:
            meta[field_name] = value
    return Candidate(meta=meta, skill=skill, fitness_fn=fitness_fn)



def _ask(client: LLMClient, prompt: Dict[str, str], temperature=None) -> Dict:
    """One model round-trip -> extracted, explanation-checked JSON object.

    A completion that will not parse is written to ``cache/unparseable/`` before the
    exception propagates -- four Phase-2 runs died on the same parse error with
    nothing on disk but the exception text, so every fix was a guess at what the
    model had actually sent.
    """
    raw = client.complete(prompt["system"], prompt["user"], temperature=temperature)
    try:
        return extract_json(raw)
    except Exception:
        dump_unparseable(
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
    fixed_fitness_fn=None,
    fixed_fitness_code: Optional[str] = None,
    fixed_objective: Optional[str] = None,
    require_mechanism: bool = False,
    require_self_check: bool = False,
    n_repair: int = 2,
    temperature: Optional[float] = None,
) -> Candidate:
    """Call the model; on a validation error, feed it back and retry ``n_repair``
    times. Raises :class:`EvolutionError` if no attempt validates.

    Each retry asks COOLER (:func:`~pref_dispatch.llm.repair.repair_temperature`).
    Retrying at the same high temperature just re-rolls the same dice, and malformed
    output is exactly the failure mode high temperature causes -- diversity is worth
    paying for in the first attempt, not in the recovery.

    ``fixed_*``, ``require_mechanism`` and ``require_self_check`` are forwarded to
    :func:`_build_candidate`.
    """
    feedback: Optional[str] = None
    last_err = ""
    for attempt in range(n_repair + 1):
        prompt = build_prompt(feedback)
        try:
            obj = _ask(client, prompt,
                       temperature=repair_temperature(temperature, attempt))
            return _build_candidate(
                obj, gen,
                fixed_fitness_fn=fixed_fitness_fn,
                fixed_fitness_code=fixed_fitness_code,
                fixed_objective=fixed_objective,
                require_mechanism=require_mechanism,
                require_self_check=require_self_check,
            )
        except (SandboxError, Exception) as e:  # noqa: BLE001 - retry on any
            last_err = f"{type(e).__name__}: {e}"
            feedback = last_err
    raise EvolutionError(f"no valid candidate after repairs; last error: {last_err}")


def _seed_candidate(
    seed_code: str,
    seed_fitness_code: str,
    seed_objective: str,
    seed_meta: Optional[Dict] = None,
) -> Candidate:
    """Warm-start incumbent: compile + validate a frozen skill's code + its
    self-authored fitness as gen-0. No LLM call.

    The frozen skill's ``fitness_code`` becomes the FIXED yardstick for the whole
    fine-tune run (same paradigm as a fresh run: fitness is set once, then only the
    score body is hill-climbed). ``seed_meta`` supplies name/description/rationale;
    missing NL fields fall back to placeholders so the contract check passes.
    """
    sm = seed_meta or {}
    obj = {
        "skill_name": str(sm.get("skill_name", "seed_skill")).strip() or "seed_skill",
        "objective": str(seed_objective or sm.get("objective", "warm-start seed")).strip()
        or "warm-start seed",
        "description": str(
            sm.get("description", "Frozen skill seeded for fine-tuning.")
        ).strip()
        or "Frozen skill seeded for fine-tuning.",
        "fitness_code": seed_fitness_code,
        "fitness_rationale": str(
            sm.get("fitness_rationale", "Inherited from the frozen seed skill.")
        ).strip()
        or "Inherited from the frozen seed skill.",
        "code": seed_code,
    }
    # fixed_fitness_fn=None so _build_candidate compiles the seed's fitness_code as
    # the fixed yardstick for the run.
    return _build_candidate(obj, gen=0, fixed_fitness_fn=None)


def evolve_one_skill(
    client: LLMClient,
    env_profile: str,
    *,
    objective_hint: Optional[str] = None,
    existing_skills: Optional[Sequence[Dict]] = None,
    reference: Optional[Skill] = None,
    similarity_note: Optional[str] = None,
    repository_note: Optional[str] = None,
    scenarios: Optional[Sequence] = None,
    reference_scenario_metrics: Optional[Sequence[Dict]] = None,
    rescale: str = "reference",
    seed_code: Optional[str] = None,
    seed_fitness_code: Optional[str] = None,
    seed_objective: Optional[str] = None,
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
) -> Candidate:
    """Evolve a single skill and return the best :class:`Candidate`.

    Two evaluation modes:

    * **v2 (scenarios given)** -- every candidate is scored by
      :func:`evaluate_skill_random_scenarios` over the SAME batch of randomized
      scenarios (fleet/capacity/speed/regime/preference), so the skill is selected
      for GENERALIZATION, not for overfitting one operating point. ``reference``
      (rolled once per scenario, or precomputed ``reference_scenario_metrics``)
      gives the scale-free per-scenario Δ under ``rescale="reference"``.
    * **v1 (scenarios None)** -- the legacy fixed-point evaluation
      (:func:`evaluate_skill` over ``regimes`` at ``num_drivers``); ``reference``
      is the handwritten seed used as the scale-free baseline. Kept for regression.

    ``similarity_note`` is injected into the proposal prompt EVERY round (not only
    after a rejection) so the model always sees how close it is drifting to the
    skills already in the basis and is pushed toward an uncovered niche.
    ``repository_note`` (v3) adds the repository's own state -- every member with
    its measured redundancy, plus the replacement rule once the repository is full
    -- so the proposal is written against the crowding it must actually beat.

    §Phase-3 warm-start fine-tune (``seed_code`` + ``seed_fitness_code`` given):
    generation 0 is seeded DIRECTLY from an already-frozen skill's ``code`` and its
    self-authored ``fitness_code`` (the fixed yardstick), with NO LLM call. The
    frozen fitness becomes the run's fixed yardstick and generations 1..G hill-climb
    the score body on top of it -- specialising a generalist skill to one concrete
    scenario at minimal LLM cost. ``seed_code=None`` (default) is unchanged.
    """
    existing = list(existing_skills or [])
    use_scenarios = scenarios is not None

    # Roll the reference ONCE (deterministic for fixed seeds) so every candidate's
    # delta reuses it instead of re-rolling the baseline.
    ref_metrics = None       # v1: {regime: metrics}
    ref_sc_metrics = None    # v2: [metrics per scenario]
    if use_scenarios:
        if reference_scenario_metrics is not None:
            ref_sc_metrics = list(reference_scenario_metrics)
        elif reference is not None:
            ref_sc_metrics = reference_metrics_for_scenarios(reference, scenarios)
    elif reference is not None:
        ref_metrics = reference_metrics_for(
            reference, regimes=regimes, split=split,
            num_drivers=num_drivers, order_limit=order_limit, seed=seed,
        )

    def _evaluate(skill, fitness_fn):
        if use_scenarios:
            return evaluate_skill_random_scenarios(
                skill, fitness_fn, scenarios,
                rescale=rescale, reference_metrics=ref_sc_metrics,
            )
        return evaluate_skill(
            skill, fitness_fn, regimes=regimes, split=split,
            num_drivers=num_drivers, order_limit=order_limit, seed=seed,
            reference_metrics=ref_metrics,
        )

    # --- Generation 0: propose (or warm-start from a frozen skill). -------- #
    if seed_code is not None:
        if not seed_fitness_code:
            raise EvolutionError(
                "warm-start requires seed_fitness_code (the frozen skill's fitness "
                "yardstick to hill-climb under)"
            )
        best = _seed_candidate(seed_code, seed_fitness_code, seed_objective, seed_meta)
        best.evaluation = _evaluate(best.skill, best.fitness_fn)
        fixed_fitness = best.fitness_fn  # frozen yardstick for the run
        log(
            f"[gen 0] WARM-START {best.name!r} objective={best.meta['objective']!r} "
            f"score={eval_report_value(best.evaluation):.4g} "
            f"select={best.score_value:.4g}"
        )
    else:
        best = _propose_with_repair(
            client,
            lambda fb: build_skill_prompt(
                env_profile,
                objective_hint=objective_hint,
                existing_skills=existing,
                similarity_note=similarity_note,
                repository_note=repository_note,
                repair_feedback=fb,
            ),
            gen=0,
            require_self_check=True,
            temperature=temperature,
        )
        best.evaluation = _evaluate(best.skill, best.fitness_fn)
        fixed_fitness = best.fitness_fn  # frozen yardstick for the run
        log(
            f"[gen 0] {best.name!r} objective={best.meta['objective']!r} "
            f"score={eval_report_value(best.evaluation):.4g} "
            f"select={best.score_value:.4g}"
        )

    # --- Generations 1..G: improve the code under the fixed fitness. ------- #
    for gen in range(1, generations + 1):
        for _ in range(lam):
            try:
                cand = _propose_with_repair(
                    client,
                    lambda fb: build_skill_improve_prompt(
                        env_profile,
                        objective=best.meta["objective"],
                        fitness_code=best.meta["fitness_code"],
                        current_code=best.meta["code"],
                        current_fitness=eval_report_value(best.evaluation),
                        existing_skills=existing,
                        repair_feedback=fb,
                    ),
                    gen=gen,
                    fixed_fitness_fn=fixed_fitness,
                    temperature=temperature,
                )
            except EvolutionError as e:
                log(f"[gen {gen}] candidate failed: {e}")
                continue
            cand.evaluation = _evaluate(cand.skill, cand.fitness_fn)
            improved = eval_report_value(cand.evaluation) > eval_report_value(best.evaluation)
            log(
                f"[gen {gen}] {cand.name!r} score={eval_report_value(cand.evaluation):.4g}"
                f" ({'ACCEPT' if improved else 'reject'})"
            )
            if improved:
                best = cand

    return best


def freeze_skill(cand: Candidate, out_dir: str = FrozenDir, *, regime: str = "multi") -> str:
    """Write the winning skill to disk as a runnable module + meta.json.

    Returns the path to the frozen ``.py`` module. The module is import-safe and
    self-contained (it re-declares the primitives it needs so the frozen product
    has no hidden coupling to the evolution harness).
    """
    os.makedirs(out_dir, exist_ok=True)
    name = cand.name
    py_path = os.path.join(out_dir, f"{name}.py")
    meta_path = os.path.join(out_dir, f"{name}.meta.json")

    header = (
        f'"""Frozen evolved skill: {name}\n\n'
        f"Objective: {cand.meta['objective']}\n\n"
        + (f"Mechanism: {cand.meta['mechanism']}\n\n"
           if cand.meta.get("mechanism") else "")
        + (f"Objective self-check: {cand.meta['objective_self_check']}\n\n"
           if cand.meta.get("objective_self_check") else "")
        + f"{cand.meta['description']}\n\n"
        f"Fitness rationale: {cand.meta['fitness_rationale']}\n"
        f"Generated in gen {cand.meta['gen']} (regime={regime}). Paradigm B: this\n"
        f'runs at ~zero online LLM cost."""\n\n'
        "import math\n"
        "import numpy as np\n\n"
        "from pref_dispatch.skills import (\n"
        "    _feasible, _pickup_time, _solo_time, _onboard_slack,\n"
        ")\n\n\n"
    )
    with open(py_path, "w", encoding="utf-8") as f:
        f.write(header + cand.meta["code"].rstrip() + "\n")

    meta = dict(cand.meta)
    meta["regime"] = regime
    ev = cand.evaluation
    # Duck-typed, deliberately: the group loop's SkillGroupEval lives in
    # evolve_skill_group, which imports THIS module, so naming its class here would
    # be a cycle. Its distinguishing field is the per-scenario advantage row.
    if ev is not None and hasattr(ev, "per_scenario_adv"):
        meta["eval_mode"] = "group"
        meta["fitness"] = ev.fitness
        meta["raw_fitness"] = ev.raw_fitness
        meta["per_scenario_adv"] = list(ev.per_scenario_adv)
        meta["per_scenario_raw"] = list(ev.per_scenario_raw)
        meta["per_band"] = dict(ev.per_band)
        meta["invalid_rate"] = ev.invalid_rate
        meta["scenario_labels"] = list(ev.labels)
    elif isinstance(ev, ScenarioEval):
        meta["eval_mode"] = "scenarios"
        meta["score"] = ev.score
        meta["per_scenario_score"] = ev.per_scenario_score
        meta["scenario_labels"] = ev.labels
    elif ev is not None:
        meta["eval_mode"] = "fixed_point"
        meta["fitness_mean"] = ev.fitness_mean
        meta["delta_vs_reference"] = ev.delta_vs_reference
        meta["per_regime_fitness"] = ev.per_regime_fitness
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return py_path


def discard_frozen_skill(py_path: str, *, subdir: str = "discarded") -> str:
    """Move an evicted skill's frozen ``.py`` + ``.meta.json`` out of the load path.

    :func:`pref_dispatch.llm.basis.load_basis` globs ``*.meta.json`` FLAT in the
    frozen directory, so a skill that Phase-1 replacement evicted would still be
    loaded at inference if its files stayed put. Relocating the pair into a
    subdirectory removes it from the basis while keeping it on disk for audit.
    Returns the new ``.py`` path (or the original if it no longer exists).
    """
    if not os.path.exists(py_path):
        return py_path
    src_dir, py_name = os.path.split(py_path)
    dst_dir = os.path.join(src_dir, subdir)
    os.makedirs(dst_dir, exist_ok=True)
    stem = os.path.splitext(py_name)[0]
    moved = os.path.join(dst_dir, py_name)
    for src, dst in (
        (py_path, moved),
        (os.path.join(src_dir, f"{stem}.meta.json"),
         os.path.join(dst_dir, f"{stem}.meta.json")),
    ):
        if os.path.exists(src):
            os.replace(src, dst)
    return moved
