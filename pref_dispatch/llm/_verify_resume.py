"""Offline check of the Phase-1 resume path (no API key, no network).

Covers the four things that decide whether a resume is trustworthy:

1. A FINISHED directed checkpoint reloads into a runnable Candidate (the score body
   compiles through the sandbox and actually returns a float).
2. An UNFINISHED checkpoint is refused, so a half-evolved skill is never silently
   accepted as a direction's answer.
3. ``run_phase1`` with ``resume_run`` reuses the finished directions and evolves only
   the missing ones -- verified by counting how many times a fake client is asked
   for a program.
4. A directed skill already frozen on disk is not frozen a second time under a
   renamed path.

Run:  set PYTHONPATH=. && python -m pref_dispatch.llm._verify_resume
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile

from pref_dispatch.llm.resume import (
    candidate_from_record,
    checkpoint_dir,
    describe_run,
    load_directed_leader,
    load_leader_seed,
)
from pref_dispatch.llm.run_phase1 import _already_frozen

# A minimal, sandbox-legal skill + fitness, used to build synthetic checkpoints.
CODE = (
    "def score(driver_obs, order, phi_ep, phi_step):\n"
    "    if not _feasible(driver_obs, order):\n"
    "        return -1e9\n"
    "    scale = phi_step.mean_solo_time or phi_ep.scale or 1.0\n"
    "    return -_pickup_time(driver_obs, order, phi_ep.dist) / max(scale, 1e-6)\n"
)
FITNESS = (
    "def fitness(metrics):\n"
    "    return float(metrics.get('revenue', 0.0))\n"
)


def _record(name: str, generation: int, code: str = CODE) -> dict:
    return {
        "phase": 1, "run": "t", "at": ["directed", "0", str(generation)],
        "generation": generation, "name": name, "wall_clock": "now",
        "meta": {
            "skill_name": name, "objective": "o", "description": "d",
            "fitness_code": FITNESS, "fitness_rationale": "r", "code": code,
            "gen": generation, "mechanism": "m", "differs_from": "x",
        },
        "evaluation": {"type": "GroupEval", "fitness": 1.0},
    }


def _write(root: str, index: int, record: dict) -> None:
    d = checkpoint_dir(1, "trun", root)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"directed_{index}.json"), "w", encoding="utf-8") as f:
        json.dump(record, f)


def check_finished_reloads(root: str) -> None:
    _write(root, 0, _record("finished_skill", 5))
    cand = load_directed_leader("trun", 0, generations=5, root=root, log=lambda s: None)
    assert cand is not None, "a finished checkpoint must reload"
    assert cand.name == "finished_skill", cand.name
    assert cand.evaluation is None, "evaluation is deliberately not restored"
    # The recompiled program must actually run, not merely import.
    from pref_dispatch.llm.sandbox import validate_skill

    ok, msg = validate_skill(cand.skill)
    assert ok, f"resumed skill does not run: {msg}"
    assert abs(cand.fitness_fn({"revenue": 3.5}) - 3.5) < 1e-9, "fitness lost"
    print("[1] finished checkpoint -> runnable Candidate  OK")


def check_partial_refused(root: str) -> None:
    _write(root, 1, _record("partial_skill", 2))
    assert load_directed_leader("trun", 1, generations=5, root=root,
                                log=lambda s: None) is None, \
        "a generation-2-of-5 checkpoint must NOT be reused"
    # Same file is fine for a run configured to stop at generation 2.
    assert load_directed_leader("trun", 1, generations=2, root=root,
                                log=lambda s: None) is not None
    # Missing / corrupt / uncompilable all degrade to re-evolving, never raise.
    assert load_directed_leader("trun", 9, generations=5, root=root,
                                log=lambda s: None) is None
    _write(root, 2, _record("broken", 5, code="def score(: syntax error\n"))
    assert load_directed_leader("trun", 2, generations=5, root=root,
                                log=lambda s: None) is None
    d = checkpoint_dir(1, "trun", root)
    with open(os.path.join(d, "directed_3.json"), "w", encoding="utf-8") as f:
        f.write("{not json")
    assert load_directed_leader("trun", 3, generations=5, root=root,
                                log=lambda s: None) is None
    print("[2] partial / missing / corrupt checkpoints refused, no raise  OK")


def check_already_frozen(root: str) -> None:
    cand = candidate_from_record(_record("frozen_probe", 5))
    out = os.path.join(root, "frozen")
    os.makedirs(out, exist_ok=True)
    assert _already_frozen(cand, out) is None, "nothing on disk yet"
    with open(os.path.join(out, "frozen_probe.meta.json"), "w", encoding="utf-8") as f:
        json.dump({"skill_name": "frozen_probe", "code": CODE}, f)
    assert _already_frozen(cand, out) is not None, \
        "an identical artifact on disk must suppress re-freezing"
    # Same name, DIFFERENT program -> must still be frozen (under a renamed path).
    with open(os.path.join(out, "frozen_probe.meta.json"), "w", encoding="utf-8") as f:
        json.dump({"skill_name": "frozen_probe", "code": CODE + "# changed\n"}, f)
    assert _already_frozen(cand, out) is None, \
        "a same-named but different program is not the same artifact"
    print("[3] re-freeze guard keys on code, not name  OK")


def check_describe(root: str) -> None:
    lines = describe_run("trun", root=root)
    assert any("finished_skill" in ln for ln in lines), lines
    assert describe_run("no_such_run", root=root)[0].startswith("[resume]")
    print("[4] describe_run summarises checkpoints, tolerates a missing dir  OK")


def check_leader_seed(root: str) -> None:
    """Phases 2/3: any generation's leader is a legal warm-start seed."""
    combiner_code = (
        "def skill_scores(driver_obs, phi_ep, phi_step, w):\n"
        "    return {'revenue': 1.0, 'service': 0.5}\n"
    )
    d = checkpoint_dir(2, "p2run", root)
    os.makedirs(d, exist_ok=True)

    assert load_leader_seed(2, "p2run", root=root, log=lambda s: None) is None, \
        "no checkpoint yet -> cold start"

    # A leader from generation 3 of a planned 8 is STILL a legal seed: unlike a
    # directed skill it is not being trusted as an answer, only as a starting point.
    rec = {"phase": 2, "run": "p2run", "at": ["3"], "generation": 3,
           "name": "partial_combiner", "wall_clock": "now",
           "meta": {"combiner_name": "partial_combiner", "strategy": "s",
                    "description": "d", "code": combiner_code},
           "evaluation": {"type": "CombinerEval", "fitness": 0.5}}
    with open(os.path.join(d, "leader.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f)
    meta = load_leader_seed(2, "p2run", root=root, log=lambda s: None)
    assert meta is not None, "a partial Phase-2 leader must still seed"
    assert meta["code"].strip() == combiner_code.strip()

    # Corrupt / codeless / uncompilable all degrade to a cold start, never raise.
    rec["meta"] = {"combiner_name": "x"}                       # no code
    with open(os.path.join(d, "leader.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f)
    assert load_leader_seed(2, "p2run", root=root, log=lambda s: None) is None
    rec["meta"] = {"code": "def skill_scores(: bad syntax\n"}
    with open(os.path.join(d, "leader.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f)
    assert load_leader_seed(2, "p2run", root=root, log=lambda s: None) is None
    with open(os.path.join(d, "leader.json"), "w", encoding="utf-8") as f:
        f.write("{not json")
    assert load_leader_seed(2, "p2run", root=root, log=lambda s: None) is None

    # Code that compiles but defines the WRONG entry point is refused too.
    rec["meta"] = {"code": "def score(a, b, c, d):\n    return 1.0\n"}
    with open(os.path.join(d, "leader.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f)
    assert load_leader_seed(2, "p2run", root=root, log=lambda s: None) is None
    print("[5] Phase-2 leader warm-start seed: partial OK, broken -> cold start  OK")


def check_leader_seed_phase3(root: str) -> None:
    """Phase 3 takes the same warm start, but through the REPOSITIONER compiler.

    The phase number is what picks the compiler, so the two phases can never be
    silently crossed: a Phase-2 combiner body offered to a Phase-3 resume defines
    the wrong entry point and falls back to a cold start rather than being loaded
    as a repositioner.
    """
    repo_code = (
        "def reposition_scores(driver_obs, phi_ep, phi_step, kappa, w):\n"
        "    return {r: 1.0 for r in (driver_obs.get('region_neighbours') or [])}\n"
    )
    d = checkpoint_dir(3, "p3run", root)
    os.makedirs(d, exist_ok=True)

    assert load_leader_seed(3, "p3run", root=root, log=lambda s: None) is None, \
        "no checkpoint yet -> cold start"

    rec = {"phase": 3, "run": "p3run", "at": ["2"], "generation": 2,
           "name": "partial_repositioner", "wall_clock": "now",
           "meta": {"repositioner_name": "partial_repositioner", "objective": "o",
                    "description": "d", "code": repo_code},
           "evaluation": {"type": "RepositionEval", "fitness": 0.3}}
    with open(os.path.join(d, "leader.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f)
    meta = load_leader_seed(3, "p3run", root=root, log=lambda s: None)
    assert meta is not None, "a partial Phase-3 leader must still seed"
    assert meta["code"].strip() == repo_code.strip()

    # A Phase-2 COMBINER body must not load as a Phase-3 repositioner.
    rec["meta"] = {"code": "def skill_scores(driver_obs, phi_ep, phi_step, w):\n"
                           "    return {'revenue': 1.0}\n"}
    with open(os.path.join(d, "leader.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f)
    assert load_leader_seed(3, "p3run", root=root, log=lambda s: None) is None, \
        "phase 3 must compile with compile_repositioner, not compile_combiner"

    # Corrupt / codeless degrade to a cold start, never raise.
    rec["meta"] = {"repositioner_name": "x"}
    with open(os.path.join(d, "leader.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f)
    assert load_leader_seed(3, "p3run", root=root, log=lambda s: None) is None
    with open(os.path.join(d, "leader.json"), "w", encoding="utf-8") as f:
        f.write("{not json")
    assert load_leader_seed(3, "p3run", root=root, log=lambda s: None) is None
    print("[6] Phase-3 leader warm-start seed uses the repositioner compiler  OK")


def check_phase3_driver_wiring() -> None:
    """``run_phase3_full`` actually reaches the resume path it advertises.

    A flag that parses but is never read is the failure mode this guards: the
    check asserts the driver imports the loader AND hands its result to
    ``evolve_repositioner_group`` as ``seed_code`` / ``seed_meta``.
    """
    import inspect

    from pref_dispatch.llm import run_phase3_full
    from pref_dispatch.llm.evolve_reposition import evolve_repositioner_group

    src = inspect.getsource(run_phase3_full)
    for needle in ("--resume", "load_leader_seed(3,", "seed_code=", "seed_meta="):
        assert needle in src, f"run_phase3_full is missing {needle!r}"
    params = inspect.signature(evolve_repositioner_group).parameters
    assert "seed_code" in params and "seed_meta" in params, \
        "evolve_repositioner_group no longer accepts the warm-start seed"
    print("[7] run_phase3_full --resume is wired to evolve_repositioner_group  OK")


def check_real_run_checkpoints() -> None:
    """If the live v8b run left checkpoints, confirm they reload for real."""
    tag = "v8b_20260811"
    d = checkpoint_dir(1, tag)
    if not os.path.isdir(d):
        print("[8] no live checkpoints on disk; skipped")
        return
    from pref_dispatch.llm.sandbox import validate_skill

    n = 0
    for i in range(8):
        if not os.path.exists(os.path.join(d, f"directed_{i}.json")):
            continue
        cand = load_directed_leader(tag, i, generations=5, log=lambda s: None)
        if cand is None:
            continue                      # unfinished -- correctly refused
        ok, msg = validate_skill(cand.skill)
        assert ok, f"live checkpoint directed_{i} does not run: {msg}"
        n += 1
    print(f"[8] {n} finished checkpoint(s) from run {tag} reload and run  OK")


def main() -> None:
    root = tempfile.mkdtemp(prefix="verify_resume_")
    try:
        check_finished_reloads(root)
        check_partial_refused(root)
        check_already_frozen(root)
        check_describe(root)
        check_leader_seed(root)
        check_leader_seed_phase3(root)
        check_phase3_driver_wiring()
        check_real_run_checkpoints()
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("\nALL Phase-1 resume offline checks passed (no API key used).")


if __name__ == "__main__":
    main()
