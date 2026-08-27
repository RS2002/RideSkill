"""Run the Phase-2/3 rollouts on several cores instead of one.

A round of the v6 (mu+lambda) search is ``(mu + lambda) x len(batch)`` full real
demand hours -- around 160 rollouts at 7-17 s each, i.e. 20-45 minutes of wall
clock on one core while 27 others idle. Nothing in a rollout depends on any other
rollout, so this module fans them out over processes.

Threads would not help: a rollout is Python-level scoring loops, so the GIL keeps
it single-core. Processes need every task argument to survive pickling, and two of
the interesting objects do NOT -- the combiner and the LLM-authored reward are
functions produced by :mod:`pref_dispatch.llm.sandbox`'s ``exec``, which pickle
cannot name. So a task ships the SOURCE plus the parameters, and the worker
recompiles through the same sandbox validation the parent used. Concretely a task
carries:

* the combiner's ``code`` + its skill names / blend parameters,
* the :class:`~pref_dispatch.scenario.Scenario` (a dataclass of scalars),
* an OBJECTIVE PAYLOAD: either the reward dataclass itself (the key-free families
  pickle fine) or the authored reward's source, recompiled worker-side.

The frozen skill basis travels alongside, once per task but memoized worker-side,
so a worker rebuilds it once per run: sandbox-compiled skills as source, frozen
evolved skills as the path of the ``.py`` they were loaded from, handwritten seeds
as themselves (see :func:`skills_payload`).

Determinism is unchanged: every rollout is seeded by its own scenario, so the
result of a (candidate, pair) task does not depend on which process ran it or in
what order. ``workers <= 1`` uses the plain in-process loop, byte-identical to the
sequential path -- that is what the offline tests and any hand-written-skill
caller keep using.

One real difference, and the reason it does not bite: the sequential loop reuses
ONE combiner object across a candidate's pairs, while a worker builds a fresh one
per pair. A combiner that hid state in module globals would therefore score
differently on the two paths. Authored combiners are pure ``skill_scores``
functions -- module state is what the fallback rule eliminates, not something the
search rewards -- so the offline check
(``_verify_partB.test_b2j_parallel_matches_sequential``) demands exact equality of
the reward rows, not closeness.

Phase 1 fans out the same way but over a different unit of work: there the pool is
the (mu+lambda) variants of ONE skill and each task is a single-skill rollout, so
:func:`parallel_skill_group_rows` ships ALL of a round's variants as one basis
payload -- see its docstring for why one-payload-per-round is the only shape the
worker-side memo can serve.
"""

from __future__ import annotations

import math
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from pref_dispatch.matching import DEFAULT_BLEND_K

# --------------------------------------------------------------------------- #
# Payloads: what crosses the process boundary.                                 #
# --------------------------------------------------------------------------- #


class NotParallelizable(RuntimeError):
    """A task could not be described in a picklable way; caller runs sequentially."""


def skills_payload(skills: Dict) -> List[Dict]:
    """Describe the frozen basis so a worker can rebuild it, in order.

    Three kinds of skill reach this function, and each travels differently:

    * a :class:`~pref_dispatch.llm.sandbox.CompiledSkill` (what Phase 1 evolves
      in memory) is a sandbox-``exec``'d function -- unpicklable -- but since v6 it
      keeps the source it was compiled from, so it travels as SOURCE and is
      recompiled worker-side by the same validator;
    * a FROZEN evolved skill loaded by :func:`pref_dispatch.llm.basis.load_basis`
      is a module built at runtime from a ``.py`` on disk -- pickle can name
      neither the wrapper class nor the functions, and its own imports mean the
      restricted exec cannot take the source either -- so its PATH travels and the
      worker reloads the same file;
    * a handwritten :class:`~pref_dispatch.skills.Skill` is an ordinary
      module-level object and travels as itself.
    """
    out: List[Dict] = []
    for name, sk in skills.items():
        code = getattr(sk, "code", None)
        if isinstance(code, str) and code.strip():
            out.append({"name": name, "kind": "code", "code": code})
            continue
        path = getattr(sk, "source_path", None)
        if isinstance(path, str) and os.path.exists(path):
            out.append({"name": name, "kind": "path", "path": path})
            continue
        try:
            pickle.dumps(sk)
        except Exception as e:  # noqa: BLE001
            raise NotParallelizable(
                f"skill {name!r} is neither picklable nor carries its source "
                f"({type(e).__name__})") from e
        out.append({"name": name, "kind": "obj", "skill": sk})
    return out


def objective_payload(obj) -> Dict:
    """Describe one sampled objective so a worker can rebuild its reward.

    Authored rewards (the ``nl`` family, and ``weights`` with a client) are
    sandbox-compiled closures -- unpicklable -- but they carry their own source in
    ``meta["code"]``, so they travel as source and are recompiled worker-side
    through :func:`~pref_dispatch.llm.sandbox.compile_reward`. The key-free
    families are plain dataclasses and travel as themselves.
    """
    rf = getattr(obj, "reward_function", None)
    if rf is None:
        return {"kind": "none"}
    code = (getattr(obj, "meta", None) or {}).get("code")
    if code:
        return {"kind": "code", "code": code}
    try:
        pickle.dumps(rf)
    except Exception as e:  # noqa: BLE001
        raise NotParallelizable(
            f"objective {getattr(obj, 'label', obj)!r} has neither picklable "
            f"reward nor source code ({type(e).__name__})") from e
    return {"kind": "obj", "reward_function": rf}


def combiner_payload(candidate, *, soft: bool = False, blend_k: int = DEFAULT_BLEND_K) -> Dict:
    """Describe one combiner candidate as source + construction parameters."""
    code = (candidate.meta or {}).get("code")
    if not code:
        raise NotParallelizable(
            f"combiner {getattr(candidate, 'name', '?')!r} carries no source code")
    return {
        "code": code,
        "skill_names": list(candidate.skill_names),
        "soft": bool(soft),
        "blend_k": int(blend_k),
    }


# --------------------------------------------------------------------------- #
# Worker side.                                                                 #
# --------------------------------------------------------------------------- #
_SKILL_CACHE: Dict = {}
_REWARD_CACHE: Dict[str, object] = {}


def _worker_skills(payload: Sequence[Dict]) -> Dict:
    """Rebuild the frozen basis in this process, cached across tasks.

    Recompiling every skill for every task would cost more than the rollout, so
    the compiled basis is memoized on the sources it came from -- one rebuild per
    worker per run, whatever the task count."""
    key = tuple((p["name"], p.get("code") or p.get("path") or "obj")
                for p in payload)
    if _SKILL_CACHE.get("key") == key:
        return _SKILL_CACHE["basis"]
    from pref_dispatch.llm.basis import _load_evolved_module
    from pref_dispatch.llm.sandbox import compile_skill

    basis = {}
    for p in payload:
        if p["kind"] == "code":
            basis[p["name"]] = compile_skill(p["code"], name=p["name"])
        elif p["kind"] == "path":
            basis[p["name"]] = _load_evolved_module(p["path"], p["name"])
        else:
            basis[p["name"]] = p["skill"]
    _SKILL_CACHE["key"] = key
    _SKILL_CACHE["basis"] = basis
    return basis


def _worker_reward(payload: Dict):
    if payload["kind"] == "none":
        return None
    if payload["kind"] == "obj":
        return payload["reward_function"]
    code = payload["code"]
    if code not in _REWARD_CACHE:
        from pref_dispatch.llm.evolve_reward import _adapt_event_reward
        from pref_dispatch.llm.sandbox import compile_reward

        _REWARD_CACHE[code] = _adapt_event_reward(compile_reward(code))
    return _REWARD_CACHE[code]


def _run_pair(task: Dict) -> Dict:
    """One (combiner, scenario, objective) rollout, start to finish, in a worker.

    Returns the pair's reward plus this rollout's own telemetry. Fallback counts
    are summed by the parent across the row, exactly as the sequential path
    accumulates them on the shared :class:`LLMCombiner` instance.
    """
    from pref_dispatch.llm.combiner_adapter import LLMCombiner
    from pref_dispatch.llm.combiner_eval import REWARD_METRIC, _event_w
    from pref_dispatch.llm.sandbox import compile_combiner
    from pref_dispatch.evaluate import DispatchController, rollout
    from pref_dispatch.scenario import build_env

    cp = task["combiner"]
    skills = _worker_skills(task["skills"])
    combiner = LLMCombiner(
        compile_combiner(cp["code"]), cp["skill_names"],
        soft=cp["soft"], blend_k=cp["blend_k"],
    )
    if task.get("capture"):
        combiner.enable_capture(int(task["capture"]))

    reward_function = _worker_reward(task["objective"])
    sc = task["scenario"]
    env = build_env(sc, reward_function=reward_function)
    ctrl = DispatchController(combiner, skills=skills)
    metrics = rollout(env, ctrl, sc.preference, seed=sc.seed,
                      reward_fn=_event_w(reward_function))

    out = {
        "cand": task["cand"],
        "pair": task["pair"],
        "reward": float(metrics[REWARD_METRIC]),
        "n_calls": combiner.n_calls,
        "n_fallbacks": combiner.n_fallbacks,
        "reason": combiner.first_fallback_reason,
        "picks": None,
    }
    # Blindness is report-only, but it needs the driver sample this rollout saw.
    # Rather than ship 400 observation triples back, the worker replays them here
    # against the round's distinct objectives and returns the resulting fleet
    # mixes -- a handful of floats.
    grid = task.get("grid")
    if grid:
        picks = []
        for gp in grid:
            rf = _worker_reward(gp)
            d = combiner.fleet_pick_fractions(_event_w(rf))
            if d:
                picks.append(d)
        out["picks"] = picks
    return out


def _run_skill_pair(task: Dict) -> Dict:
    """One (single frozen skill, scenario, objective) rollout in a worker."""
    from pref_dispatch.llm.combiner_eval import REWARD_METRIC, _event_w
    from pref_dispatch.combiner import SingleSkillCombiner
    from pref_dispatch.evaluate import DispatchController, rollout
    from pref_dispatch.scenario import build_env

    skills = _worker_skills(task["skills"])
    reward_function = _worker_reward(task["objective"])
    sc = task["scenario"]
    env = build_env(sc, reward_function=reward_function)
    ctrl = DispatchController(SingleSkillCombiner(task["skill"]), skills=skills)
    metrics = rollout(env, ctrl, sc.preference, seed=sc.seed,
                      reward_fn=_event_w(reward_function))
    return {"skill": task["skill"], "pair": task["pair"],
            "reward": float(metrics[REWARD_METRIC])}


def _run_baseline_pair(task: Dict) -> Dict:
    """One (equal-blend baseline, scenario, objective) rollout in a worker.

    The baseline the Phase-2 fitness subtracts: every frozen skill weighted the
    same, for every driver, under every objective -- "no choice was made". It is
    also exactly what a fully-broken candidate runs
    (:meth:`pref_dispatch.llm.combiner_adapter.LLMCombiner._equal_blend`), which
    is what makes such a candidate score exactly 0.0.
    """
    from pref_dispatch.llm.combiner_eval import REWARD_METRIC, _event_w
    from pref_dispatch.combiner import EqualBlendCombiner
    from pref_dispatch.evaluate import DispatchController, rollout
    from pref_dispatch.scenario import build_env

    skills = _worker_skills(task["skills"])
    reward_function = _worker_reward(task["objective"])
    sc = task["scenario"]
    env = build_env(sc, reward_function=reward_function)
    ctrl = DispatchController(
        EqualBlendCombiner(list(skills), blend_k=task["blend_k"]), skills=skills)
    metrics = rollout(env, ctrl, sc.preference, seed=sc.seed,
                      reward_fn=_event_w(reward_function))
    return {"pair": task["pair"], "reward": float(metrics[REWARD_METRIC])}


# --------------------------------------------------------------------------- #
# Phase-1 worker side: one skill's variants, each rolled alone.                #
# --------------------------------------------------------------------------- #
_GROUP_CACHE: Dict = {}


def _worker_group_skills(payload: Sequence[Dict]) -> Dict:
    """Recompile a whole Phase-1 round's variants in this process, memoized.

    Same single-slot memo as :func:`_worker_skills`, with one deliberate
    difference: a variant that fails to COMPILE is stored as the exception rather
    than raised here. A Phase-1 pool is by construction full of unproven programs,
    and one that the worker cannot rebuild must fail its own cells only -- raising
    during the rebuild would fail every other variant's cells too, and the round
    would read as "everything died" instead of "this one died".
    """
    key = tuple((p["name"], p.get("code") or p.get("path") or "obj")
                for p in payload)
    if _GROUP_CACHE.get("key") == key:
        return _GROUP_CACHE["basis"]
    from pref_dispatch.llm.sandbox import compile_skill

    basis: Dict = {}
    for p in payload:
        try:
            if p["kind"] == "code":
                basis[p["name"]] = compile_skill(p["code"], name=p["name"])
            elif p["kind"] == "path":
                from pref_dispatch.llm.basis import _load_evolved_module
                basis[p["name"]] = _load_evolved_module(p["path"], p["name"])
            else:
                basis[p["name"]] = p["skill"]
        except Exception as e:  # noqa: BLE001 -- surfaced at use, per the docstring
            basis[p["name"]] = e
    _GROUP_CACHE["key"] = key
    _GROUP_CACHE["basis"] = basis
    return basis


def _run_group_pair(task: Dict) -> Dict:
    """One (Phase-1 variant, scenario) rollout in a worker -> raw METRICS.

    The metrics dict travels home unreduced and the parent applies the fitness:
    the round's ``fitness_fn`` is a sandbox-``exec``'d closure, so it cannot be
    pickled into a task, and re-shipping its source per cell would recompile it
    once per rollout for no gain. Metrics are a handful of floats either way.
    """
    from pref_dispatch.llm.fitness_eval import rollout_skill_on_scenario

    out = {"cand": task["cand"], "pair": task["pair"],
           "metrics": None, "error": None}
    try:
        sk = _worker_group_skills(task["skills"])[task["key"]]
        if isinstance(sk, BaseException):
            raise sk
        m = rollout_skill_on_scenario(sk, task["scenario"])
        out["metrics"] = {k: float(v) for k, v in m.items()}
    except Exception as e:  # noqa: BLE001 -- a raising skill is a candidate too
        out["error"] = f"{type(e).__name__}: {e}"
    return out


# --------------------------------------------------------------------------- #
# Phase-3 worker side: one repositioner on one (scene, objective, strength).    #
# --------------------------------------------------------------------------- #
def reposition_payload(candidate) -> Dict:
    """Describe one reposition-scorer candidate as source for a worker."""
    code = (getattr(candidate, "meta", None) or {}).get("code")
    if not code:
        raise NotParallelizable(
            f"repositioner {getattr(candidate, 'name', '?')!r} carries no source code")
    return {"code": code}


def frozen_combiner_payload(code: Optional[str], skill_names: Sequence[str],
                            *, blend_k: int = DEFAULT_BLEND_K) -> Dict:
    """Describe the FROZEN Phase-2 combiner Phase 3 runs underneath.

    Phase 3 does not evolve the combiner, but every worker still needs one, and a
    compiled combiner is a sandbox closure that cannot be pickled. ``code`` is its
    ``skill_scores`` source (``load_frozen_combiner`` returns it in the meta); a
    combiner whose source is unavailable simply keeps the round in-process.
    """
    if not (isinstance(code, str) and code.strip()):
        raise NotParallelizable(
            "the frozen combiner carries no source code, so a worker cannot "
            "rebuild it")
    if not skill_names:
        raise NotParallelizable("the frozen combiner payload needs skill_names")
    return {"code": code, "skill_names": list(skill_names), "blend_k": int(blend_k)}


def _run_reposition_cell(task: Dict) -> Dict:
    """One (repositioner, scenario, objective, fairness strength) rollout.

    ``task["mode"]`` picks what sits in the repositioner slot: ``"cand"`` = the
    candidate scorer, ``"heur"`` = the built-in demand-gravity heuristic, ``"off"``
    = no repositioner at all. The two anchors run through the same worker as the
    candidates so an anchor and a candidate cell are byte-identical apart from that
    slot.
    """
    from pref_dispatch.llm.combiner_adapter import LLMCombiner
    from pref_dispatch.llm.combiner_eval import REWARD_METRIC, _event_w
    from pref_dispatch.llm.reposition_adapter import GuardedScorer
    from pref_dispatch.llm.reposition_eval import STRENGTH_PROBE_GRID, with_fairness
    from pref_dispatch.llm.sandbox import compile_combiner, compile_repositioner
    from pref_dispatch.evaluate import DispatchController, rollout
    from pref_dispatch.reposition import Repositioner
    from pref_dispatch.scenario import build_env

    skills = _worker_skills(task["skills"])
    cp = task["combiner"]
    combiner = LLMCombiner(compile_combiner(cp["code"]), cp["skill_names"],
                           blend_k=cp["blend_k"])

    mode = task["mode"]
    scorer = None
    if mode == "cand":
        scorer = GuardedScorer(compile_repositioner(task["scorer"]["code"]))
        if task.get("capture"):
            scorer.enable_capture(int(task["capture"]))

    reward_function = _worker_reward(task["objective"])
    w = _event_w(reward_function)
    sc = task["scenario"]
    env = build_env(sc, reward_function=reward_function)
    if mode == "off":
        ctrl = DispatchController(combiner, skills=skills)
    else:
        ctrl = DispatchController(
            combiner, skills=skills,
            repositioner=Repositioner(strength=1.0, scores_fn=scorer))
    metrics = rollout(env, ctrl, with_fairness(sc.preference, task["strength"]),
                      seed=sc.seed, reward_fn=w)

    out = {
        "cand": task["cand"], "cell": task["cell"],
        "reward": float(metrics[REWARD_METRIC]),
        "n_calls": 0, "n_fallbacks": 0, "n_defers": 0, "reason": None,
        "picks": None, "spicks": None,
    }
    if scorer is None:
        return out
    out["n_calls"] = scorer.n_calls
    out["n_fallbacks"] = scorer.n_fallbacks
    out["n_defers"] = scorer.n_defers
    out["reason"] = scorer.first_fallback_reason
    # Both blindness probes are report-only and both need the driver sample THIS
    # rollout saw, so they are replayed here rather than shipping 400 contexts
    # home. Objective probe: same drivers, different w. Strength probe: same
    # drivers, same w, only the fairness knob moved.
    grid = task.get("grid")
    if grid:
        picks = [d for gp in grid
                 if (d := scorer.fleet_region_fractions(_event_w(_worker_reward(gp))))]
        out["picks"] = picks
        out["spicks"] = [
            d for s in STRENGTH_PROBE_GRID
            if (d := scorer.fleet_region_fractions(w, fairness_strength=s))
        ]
    return out


# --------------------------------------------------------------------------- #
# Parent side.                                                                 #
# --------------------------------------------------------------------------- #
def resolve_workers(workers: Optional[int]) -> int:
    """How many processes to use. ``None`` = leave two cores for the OS and the
    parent; ``<= 1`` = stay in-process."""
    if workers is not None:
        return max(1, int(workers))
    return max(1, (os.cpu_count() or 2) - 2)


def _submit(pool, fn, tasks, log):
    futures = {pool.submit(fn, t): t for t in tasks}
    for fut in as_completed(futures):
        yield fut.result()


def parallel_pair_rewards(
    candidates: Sequence,
    skills: Dict,
    scenarios: Sequence,
    objectives: Sequence,
    *,
    workers: int,
    capture: int = 400,
    soft: bool = False,
    blend_k: int = DEFAULT_BLEND_K,
    log: Callable[[str], None] = print,
) -> List[Dict]:
    """Roll EVERY candidate on EVERY (scenario, objective) pair, across processes.

    Returns one dict per candidate, in the order of ``candidates``::

        {"rewards": [r_pair0, ...], "n_calls": int, "n_fallbacks": int,
         "reason": Optional[str], "picks": Optional[List[Dict]]}

    ``picks`` is the captured fleet-mix probe from that candidate's first pair
    (the blindness diagnostic); it is ``None`` when nothing was captured.

    Raises :class:`NotParallelizable` if any candidate, skill or objective cannot
    be described for a worker -- callers fall back to the sequential path rather
    than half-running a round.
    """
    obj_payloads = [objective_payload(o) for o in objectives]
    cand_payloads = [combiner_payload(c, soft=soft, blend_k=blend_k)
                     for c in candidates]
    sk_payload = skills_payload(skills)
    # The distinct-objective grid the blindness probe replays (same rule as
    # combiner_eval._objective_grid: one point per distinct w in the batch).
    grid, seen = [], set()
    for o, p in zip(objectives, obj_payloads):
        key = id(getattr(o, "w", None))
        if key not in seen:
            seen.add(key)
            grid.append(p)

    tasks = []
    for ci, cp in enumerate(cand_payloads):
        for pi, (sc, op) in enumerate(zip(scenarios, obj_payloads)):
            tasks.append({
                "cand": ci, "pair": pi, "combiner": cp, "scenario": sc,
                "objective": op, "skills": sk_payload,
                # Capture (and therefore probe) once per candidate: the sample is
                # a property of the combiner, not of which hour it ran.
                "capture": capture if pi == 0 else 0,
                "grid": grid if pi == 0 else None,
            })

    out = [{"rewards": [0.0] * len(scenarios), "n_calls": 0, "n_fallbacks": 0,
            "reason": None, "picks": None} for _ in candidates]
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for res in _submit(pool, _run_pair, tasks, log):
            rec = out[res["cand"]]
            rec["rewards"][res["pair"]] = res["reward"]
            rec["n_calls"] += res["n_calls"]
            rec["n_fallbacks"] += res["n_fallbacks"]
            if rec["reason"] is None and res["reason"]:
                rec["reason"] = res["reason"]
            if res["picks"]:
                rec["picks"] = res["picks"]
            done += 1
            if done % max(1, len(tasks) // 10) == 0:
                log(f"    [parallel] {done}/{len(tasks)} rollouts")
    return out


def parallel_skill_rows(
    skills: Dict,
    scenarios: Sequence,
    objectives: Sequence,
    *,
    workers: int,
    log: Callable[[str], None] = print,
) -> List[List[float]]:
    """Per-(scenario, objective) pair, each frozen skill rolled alone, across
    processes. Shape matches
    :func:`pref_dispatch.llm.combiner_eval._skill_reference_rewards`:
    ``refs[pair][skill_index]``."""
    obj_payloads = [objective_payload(o) for o in objectives]
    sk_payload = skills_payload(skills)
    names = list(skills)
    tasks = [
        {"skill": nm, "pair": pi, "scenario": sc, "objective": op,
         "skills": sk_payload}
        for pi, (sc, op) in enumerate(zip(scenarios, obj_payloads))
        for nm in names
    ]
    refs = [[0.0] * len(names) for _ in scenarios]
    idx = {nm: i for i, nm in enumerate(names)}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for res in _submit(pool, _run_skill_pair, tasks, log):
            refs[res["pair"]][idx[res["skill"]]] = res["reward"]
    return refs


def parallel_baseline_rows(
    skills: Dict,
    scenarios: Sequence,
    objectives: Sequence,
    *,
    workers: int,
    blend_k: int = DEFAULT_BLEND_K,
    log: Callable[[str], None] = print,
) -> List[float]:
    """Per-(scenario, objective) pair, the equal-blend "no choice was made"
    reward, across processes. Shape matches
    :func:`pref_dispatch.llm.combiner_eval._baseline_rewards`: ``base[pair]``.

    ``blend_k`` must be the CANDIDATES' value, or the baseline is not the same
    policy a fully-falling-back candidate runs and a broken program stops
    scoring exactly 0.
    """
    sk_payload = skills_payload(skills)
    tasks = [
        {"pair": pi, "scenario": sc, "objective": objective_payload(o),
         "skills": sk_payload, "blend_k": int(blend_k)}
        for pi, (sc, o) in enumerate(zip(scenarios, objectives))
    ]
    base = [0.0] * len(scenarios)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for res in _submit(pool, _run_baseline_pair, tasks, log):
            base[res["pair"]] = res["reward"]
    return base


def _objective_grid_payloads(objectives: Sequence, obj_payloads: Sequence[Dict]
                             ) -> List[Dict]:
    """One payload per DISTINCT objective in the batch (same rule as
    :func:`~pref_dispatch.llm.combiner_eval._objective_grid`)."""
    grid, seen = [], set()
    for o, p in zip(objectives, obj_payloads):
        key = id(getattr(o, "w", None))
        if key not in seen:
            seen.add(key)
            grid.append(p)
    return grid


def parallel_reposition_rows(
    candidates: Sequence,
    combiner: Dict,
    skills: Dict,
    scenarios: Sequence,
    objectives: Sequence,
    strengths: Sequence[float],
    *,
    workers: int,
    capture: int = 400,
    log: Callable[[str], None] = print,
) -> List[Dict]:
    """Roll EVERY repositioner on EVERY (scene, objective, strength) cell.

    ``combiner`` is a :func:`frozen_combiner_payload` dict -- Phase 3 does not
    evolve it, but every worker needs its own copy. Returns one dict per candidate,
    in ``candidates`` order::

        {"rewards": [...], "n_calls": int, "n_fallbacks": int, "n_defers": int,
         "reason": Optional[str], "picks": Optional[List[Dict]],
         "spicks": Optional[List[Dict]]}

    ``picks`` / ``spicks`` are the objective- and strength-blindness probes replayed
    inside the worker that holds the capture. Raises :class:`NotParallelizable` if
    anything cannot be described for a worker, so the caller falls back to the
    sequential path rather than half-running a round.
    """
    obj_payloads = [objective_payload(o) for o in objectives]
    cand_payloads = [reposition_payload(c) for c in candidates]
    sk_payload = skills_payload(skills)
    grid = _objective_grid_payloads(objectives, obj_payloads)
    # Capture on the first cell whose fairness strength is NON-ZERO when there is
    # one. A context captured at strength 0 has every budget multiplier at exactly
    # 1.0, so the income ranking behind them is gone and the strength probe can
    # only move phi_ep.fairness_strength, not the multipliers -- which reads as
    # "blind" even for a scorer that acts on them. Any positive strength keeps the
    # ranking recoverable (see reposition_adapter._rescale_budgets).
    cap_cell = next((i for i, s in enumerate(strengths) if float(s) > 0.0), 0)

    tasks = []
    for ci, cp in enumerate(cand_payloads):
        for xi, (sc, op, st) in enumerate(zip(scenarios, obj_payloads, strengths)):
            tasks.append({
                "cand": ci, "cell": xi, "mode": "cand", "scorer": cp,
                "combiner": combiner, "skills": sk_payload,
                "scenario": sc, "objective": op, "strength": float(st),
                "capture": capture if xi == cap_cell else 0,
                "grid": grid if xi == cap_cell else None,
            })

    out = [{"rewards": [0.0] * len(scenarios), "n_calls": 0, "n_fallbacks": 0,
            "n_defers": 0, "reason": None, "picks": None, "spicks": None}
           for _ in candidates]
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for res in _submit(pool, _run_reposition_cell, tasks, log):
            rec = out[res["cand"]]
            rec["rewards"][res["cell"]] = res["reward"]
            rec["n_calls"] += res["n_calls"]
            rec["n_fallbacks"] += res["n_fallbacks"]
            rec["n_defers"] += res["n_defers"]
            if rec["reason"] is None and res["reason"]:
                rec["reason"] = res["reason"]
            if res["picks"]:
                rec["picks"] = res["picks"]
            if res["spicks"]:
                rec["spicks"] = res["spicks"]
            done += 1
            if done % max(1, len(tasks) // 10) == 0:
                log(f"    [parallel] {done}/{len(tasks)} rollouts")
    return out


def parallel_reposition_anchors(
    combiner: Dict,
    skills: Dict,
    scenarios: Sequence,
    objectives: Sequence,
    strengths: Sequence[float],
    *,
    workers: int,
    log: Callable[[str], None] = print,
) -> List[List[float]]:
    """The two FIXED reference policies per cell, across processes.

    Shape matches
    :func:`~pref_dispatch.llm.reposition_eval.anchor_reference_rewards`:
    ``[[demand_gravity_heuristic, reposition_off], ...]``, one inner list per cell.
    """
    obj_payloads = [objective_payload(o) for o in objectives]
    sk_payload = skills_payload(skills)
    modes = ("heur", "off")
    tasks = [
        {"cand": mi, "cell": xi, "mode": mode, "scorer": None,
         "combiner": combiner, "skills": sk_payload,
         "scenario": sc, "objective": op, "strength": float(st),
         "capture": 0, "grid": None}
        for mi, mode in enumerate(modes)
        for xi, (sc, op, st) in enumerate(zip(scenarios, obj_payloads, strengths))
    ]
    refs = [[0.0, 0.0] for _ in scenarios]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for res in _submit(pool, _run_reposition_cell, tasks, log):
            refs[res["cell"]][res["cand"]] = res["reward"]
    return refs


def _scene_label(sc) -> str:
    fn = getattr(sc, "label", None)
    try:
        return str(fn()) if callable(fn) else "?"
    except Exception:  # noqa: BLE001
        return "?"


def skill_group_payload(pool: Sequence) -> List[Dict]:
    """Describe a whole Phase-1 round -- every variant -- as ONE basis payload.

    Keys are positional (``c0``, ``c1``, ...) rather than the candidates' own
    names: two variants of the same skill routinely carry the same ``skill_name``,
    and a name collision would silently make two rows measure one program.
    """
    out: List[Dict] = []
    for i, cand in enumerate(pool):
        code = getattr(getattr(cand, "skill", None), "code", None)
        if not (isinstance(code, str) and code.strip()):
            raise NotParallelizable(
                f"skill variant {getattr(cand, 'name', i)!r} carries no source "
                f"code, so a worker cannot rebuild it")
        out.append({"name": f"c{i}", "kind": "code", "code": code})
    return out


def parallel_skill_group_rows(
    pool: Sequence,
    scenarios: Sequence,
    *,
    workers: int,
    log: Callable[[str], None] = print,
) -> Tuple[List[List[float]], List[List[Optional[Dict[str, float]]]],
           List[Optional[str]]]:
    """Roll EVERY Phase-1 variant on EVERY scenario, across processes.

    Returns exactly what the in-process loop in
    :func:`~pref_dispatch.llm.evolve_skill_group.evolve_skill_group` returns:
    ``(rows_raw, rows_metrics, errors)``, one row per candidate in ``pool`` order,
    one column per scenario. A cell the variant raised on is ``nan`` in
    ``rows_raw`` and ``None`` in ``rows_metrics`` -- never ``0.0``, because zero is
    a legitimate fitness and scoring a crash as "average" would let a skill that
    dies on scarce fleets outrank one that merely does badly there. ``errors[i]``
    is the FIRST failure by scenario order (not by completion order, which
    ``as_completed`` scrambles), so the repair prompt sees a stable exception.

    Every variant of the round travels in ONE payload attached to every task. The
    worker's memo (:func:`_worker_group_skills`) holds a single compiled basis, so
    shipping one variant per task would evict and rebuild the basis on every
    rollout -- the memo would cost more than it saves. Built once per worker, it is
    then free for the round's whole ``(mu + lambda) x len(scenarios)`` grid.

    The fitness is applied HERE, not in the worker: it is a sandbox-compiled
    closure that cannot be pickled. Raises :class:`NotParallelizable` if any
    variant or scenario cannot be described for a worker -- callers fall back to
    the sequential path rather than half-running a round.
    """
    scs = list(scenarios)
    sk_payload = skill_group_payload(pool)
    for sc in scs:
        try:
            pickle.dumps(sc)
        except Exception as e:  # noqa: BLE001
            raise NotParallelizable(
                f"scenario {_scene_label(sc)} is not picklable "
                f"({type(e).__name__})") from e

    tasks = [
        {"cand": ci, "pair": pi, "key": f"c{ci}", "scenario": sc,
         "skills": sk_payload}
        for ci in range(len(pool))
        for pi, sc in enumerate(scs)
    ]

    rows_met: List[List[Optional[Dict[str, float]]]] = [
        [None] * len(scs) for _ in pool]
    errs: List[Optional[str]] = [None] * len(pool)
    # Which column each candidate's recorded error came from, so a later-completing
    # but earlier-numbered failure still wins.
    err_at = [len(scs)] * len(pool)
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for res in _submit(ex, _run_group_pair, tasks, log):
            ci, pi = res["cand"], res["pair"]
            if res["error"]:
                if pi < err_at[ci]:
                    err_at[ci] = pi
                    errs[ci] = f"{res['error']} (on {_scene_label(scs[pi])})"
            else:
                rows_met[ci][pi] = res["metrics"]
            done += 1
            if done % max(1, len(tasks) // 10) == 0:
                log(f"    [parallel] {done}/{len(tasks)} rollouts")

    rows_raw: List[List[float]] = []
    for ci, cand in enumerate(pool):
        raw: List[float] = []
        for pi, m in enumerate(rows_met[ci]):
            if m is None:
                raw.append(float("nan"))
                continue
            try:
                v = float(cand.fitness_fn(m))
            except Exception as e:  # noqa: BLE001 -- a fitness can raise too
                if pi < err_at[ci]:
                    err_at[ci] = pi
                    errs[ci] = (f"{type(e).__name__}: {e} "
                                f"(on {_scene_label(scs[pi])})")
                rows_met[ci][pi] = None
                raw.append(float("nan"))
                continue
            raw.append(v if math.isfinite(v) else float("nan"))
        rows_raw.append(raw)
    return rows_raw, rows_met, errs
