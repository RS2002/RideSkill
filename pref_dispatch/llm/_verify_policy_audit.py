"""Offline, key-free check of the Phase-2 / Phase-3 policy soft check (v10).

Everything here runs against a scripted client and stubbed rollouts -- no network,
no environment, no API key. What it pins down:

* the two arms of a Phase-2 cell are the SAME scene, the SAME seed and the SAME env
  reward, and differ in exactly one thing: whether ``reward_fn`` (the objective
  ``w``) was handed to the dispatcher. If that ever stops being true the contrast
  measures something other than reading the objective;
* Phase 3 adds a third arm that is ``w``-given at fairness strength 0 against the
  cell's drawn strength, and every arm carries the candidate scorer;
* the table prints the measured numbers and a percent-change column, drops columns
  that are identically zero everywhere (the reposition columns in a Phase-2 table),
  and says ``n/a`` rather than dividing by a zero baseline;
* the prompt carries the objective in the AUTHORING model's own words and carries
  NO statement of which way any column ought to move -- that is the question being
  asked, and supplying it would be us reading our own prediction back out;
* the output contract is enforced: an unknown verdict, a missing ``reason``, a
  ``per_cell`` list that does not have one entry per cell, an unknown ``moved``
  value and a note-less entry are all rejected rather than silently accepted;
* a broken audit call FAILS OPEN with ``status="error"`` and never raises -- the
  champion is already frozen and nothing downstream acts on the verdict;
* ``run_policy_audit`` says out loud, in the log, that nothing was applied.

Run: ``python -m pref_dispatch.llm._verify_policy_audit``
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from pref_dispatch.llm import policy_audit as PA
from pref_dispatch.llm.extract import ExtractionError
from pref_dispatch.preference import Preference


def _check(label: str, ok: bool, detail: str = "") -> None:
    if not ok:
        raise AssertionError(f"{label} FAILED {detail}")
    print(f"[policy-audit] {label} OK {detail}".rstrip())


# --------------------------------------------------------------------------- #
# Fixtures: a scene, an objective, and a rollout that never touches an env.    #
# --------------------------------------------------------------------------- #
@dataclass
class FakeScene:
    num_drivers: int = 400
    driver_capacity: int = 3
    speed_kmh: float = 22.0
    regime: str = "peak"
    seed: int = 4242
    clock: Tuple[Optional[str], Optional[int]] = ("Tuesday", 18)

    @property
    def preference(self) -> Preference:
        return Preference({"revenue": 0.5, "service": 0.5, "fairness": 0.0})

    def label(self) -> str:
        return f"f{self.num_drivers}_c{self.driver_capacity}_peak"


@dataclass
class FakeObjective:
    label: str = "pay only on drop-off, charge for empty driving"
    family: str = "nl"
    spec_text: str = ("+1.0 per completed drop-off; -0.20 per minute of empty "
                      "driving; nothing for merely accepting an order.")
    reward_function: object = None
    w: object = None

    def __post_init__(self) -> None:
        if self.reward_function is None:
            self.reward_function = lambda _d, _e: 1.0


#: Metrics the stub returns per arm. The w-given arm serves more and detours more;
#: the fairness-0 arm shifts the gini. Concrete values so the table is checkable.
_MET_BLIND = {"income_mean": 2.0, "assigned": 3000, "completed": 2400,
              "service_rate": 0.60, "mean_service_time": 14.0,
              "detour_total": 20000.0, "revenue": 60000.0, "income_gini": 0.16,
              "wage_gini": 0.0, "relocation_moves": 0.0,
              "reposition_distance_ratio": 0.0}
_MET_GIVEN = {**_MET_BLIND, "income_mean": 2.5, "assigned": 3300,
              "completed": 2880, "service_rate": 0.66, "detour_total": 24000.0}
_MET_FAIR0 = {**_MET_GIVEN, "income_gini": 0.22, "wage_gini": 0.31,
              "relocation_moves": 900.0, "reposition_distance_ratio": 0.12}


class RolloutSpy:
    """Stands in for ``rollout``; records every call and returns canned metrics."""

    def __init__(self) -> None:
        self.calls: List[Dict] = []

    def __call__(self, env, ctrl, pref, seed=None, *, reward_fn=None,
                 objective_label=""):
        self.calls.append({
            "env": env, "ctrl": ctrl, "seed": seed,
            "w_given": reward_fn is not None,
            "fairness": float(pref.get("fairness", 0.0)),
        })
        if reward_fn is None:
            return dict(_MET_BLIND)
        if float(pref.get("fairness", 0.0)) == 0.0 and getattr(ctrl, "rep", None):
            return dict(_MET_FAIR0)
        return dict(_MET_GIVEN)


class EnvSpy:
    def __init__(self, scene, reward_function=None):
        self.scene = scene
        self.reward_function = reward_function


class CtrlSpy:
    def __init__(self, combiner, skills=None, repositioner=None, **kw):
        self.combiner = combiner
        self.skills = skills
        self.rep = repositioner


class RepSpy:
    def __init__(self, strength=1.0, scores_fn=None):
        self.strength = strength
        self.scores_fn = scores_fn


class ScriptedAuditor:
    """Returns a canned reply per call; records every prompt it was sent."""

    def __init__(self, replies: List) -> None:
        self.replies = list(replies)
        self.prompts: List[str] = []
        self.systems: List[str] = []

    def complete(self, system: str, user: str, temperature=None) -> str:
        self.prompts.append(user)
        self.systems.append(system)
        r = self.replies[min(len(self.prompts) - 1, len(self.replies) - 1)]
        return r if isinstance(r, str) else json.dumps(r)


def _patched(fn):
    """Run ``fn(spy)`` with the env/controller/rollout stubbed out."""
    spy = RolloutSpy()
    saved = (PA.rollout, PA.build_env, PA.DispatchController, PA.Repositioner)
    PA.rollout = spy
    PA.build_env = lambda sc, reward_function=None: EnvSpy(sc, reward_function)
    PA.DispatchController = CtrlSpy
    PA.Repositioner = RepSpy
    try:
        return fn(spy)
    finally:
        (PA.rollout, PA.build_env, PA.DispatchController,
         PA.Repositioner) = saved


# --------------------------------------------------------------------------- #
# Checks.                                                                     #
# --------------------------------------------------------------------------- #
def check_phase2_arms_differ_in_one_thing() -> None:
    scenes = [FakeScene(seed=11), FakeScene(num_drivers=1200, seed=22)]
    objs = [FakeObjective(), FakeObjective(label="pay per shared seat")]

    def body(spy):
        return PA.probe_phase2(object(), {}, scenes, objs, log=lambda _s: None)

    cells = _patched(body)
    _check("one cell per (scene, objective) pair, two arms each",
           len(cells) == 2 and all(len(c.arms) == 2 for c in cells))
    _check("the arms are named for what was withheld",
           [a.name for a in cells[0].arms] == [PA.ARM_BLIND, PA.ARM_GIVEN])

    def body2(spy):
        PA.probe_phase2(object(), {}, scenes, objs, log=lambda _s: None)
        return spy.calls

    calls = _patched(body2)
    _check("two rollouts per cell", len(calls) == 4)
    a, b = calls[0], calls[1]
    _check("both arms of a cell run the SAME seed", a["seed"] == b["seed"] == 11)
    _check("both arms are graded by the SAME env reward",
           a["env"].reward_function is b["env"].reward_function
           and a["env"].reward_function is not None)
    _check("both arms replay the SAME scene", a["env"].scene is b["env"].scene)
    _check("the ONLY difference is whether w was handed over",
           a["w_given"] is False and b["w_given"] is True)
    _check("the second cell uses its own seed", calls[2]["seed"] == 22)


def check_phase3_adds_the_fairness_arm() -> None:
    scenes = [FakeScene(seed=7)]
    objs = [FakeObjective()]
    scorer = object()

    def body(spy):
        cells = PA.probe_phase3(scorer, object(), {}, scenes, objs, [0.5],
                                log=lambda _s: None)
        return cells, spy.calls

    cells, calls = _patched(body)
    _check("three arms per Phase-3 cell", len(cells[0].arms) == 3)
    _check("the third arm is the fairness-0 one",
           cells[0].arms[2].name == PA.ARM_FAIR0)
    _check("three rollouts per cell", len(calls) == 3)
    _check("arms 1-2 run at the cell's DRAWN fairness strength",
           calls[0]["fairness"] == 0.5 and calls[1]["fairness"] == 0.5)
    _check("arm 3 runs the same w-given program at fairness 0",
           calls[2]["fairness"] == 0.0 and calls[2]["w_given"] is True)
    _check("every arm carries the candidate scorer",
           all(c["ctrl"].rep is not None and c["ctrl"].rep.scores_fn is scorer
               for c in calls))
    _check("the drawn strength is printed in the scene line",
           "fairness strength 0.5" in cells[0].scene)


def check_table() -> None:
    scenes = [FakeScene(seed=11)]
    objs = [FakeObjective()]
    cells = _patched(lambda _s: PA.probe_phase2(object(), {}, scenes, objs,
                                                log=lambda _x: None))
    keys = PA._live_keys(cells)
    _check("all-zero columns are dropped from a Phase-2 table",
           "relocation_moves" not in keys and "wage_gini" not in keys
           and "reposition_distance_ratio" not in keys)
    _check("the columns that moved are kept",
           {"income_mean", "assigned", "service_rate", "detour_total"} <= set(keys))

    block = PA.contrast_table(cells)[0]
    _check("the block carries the measured numbers of both arms",
           "3,000" in block and "3,300" in block)
    _check("the change column is a percentage against the first arm",
           "+25.0%" in block and "+10.0%" in block)
    _check("the objective is shown in the authoring model's own words",
           "pay only on drop-off" in block and "-0.20 per minute" in block)
    _check("the scene is described in the block",
           "400 cars" in block and "seed 11" in block)

    _check("a zero baseline prints n/a instead of dividing",
           PA._pct(0.0, 5.0) == "n/a" and PA._pct(4.0, 5.0) == "+25.0%")

    gl = PA.column_glossary(keys)
    _check("the glossary describes exactly the columns that survived",
           "income_mean" in gl and "relocation_moves" not in gl)


def check_prompt_states_facts_only() -> None:
    scenes = [FakeScene(seed=11)]
    objs = [FakeObjective()]
    cells = _patched(lambda _s: PA.probe_phase2(object(), {}, scenes, objs,
                                                log=lambda _x: None))
    p = PA.build_policy_audit_prompt(
        phase=2,
        objective="route each driver to the skill the objective pays best for",
        description="reads w, prices a probe event, weights the skills by it",
        code="def combine(...):\n    return {}\n",
        arm_spec=PA.arm_spec(2),
        cell_blocks=PA.contrast_table(cells),
        column_glossary=PA.column_glossary(PA._live_keys(cells)),
    )
    user = p["user"]
    _check("the prompt states mechanically how the arms differ",
           "SAME random seed" in user and "was NOT given the objective" in user)
    _check("the prompt shows the program and what it claimed to do",
           "def combine" in user and "prices a probe event" in user)
    _check("the prompt says the answer is not applied to anything",
           "already frozen" in user)

    # v10 language rule: measured facts and mechanics may be stated; a direction
    # any column ought to move may not -- that is the question being asked, and
    # supplying it turns the check into us reading our own prediction back out.
    banned = ("higher better", "lower better", "should move", "we expect",
              "you should see", "the correct answer", "as intended")
    hits = [b for b in banned if b in user.lower()]
    _check("the prompt names no expected direction for any column",
           not hits, f"({hits})")
    _check("the per-cell reading is required before the verdict",
           "Fill `per_cell` FIRST" in user)


def check_contract() -> None:
    good_cell = {"cell": 1, "moved": "yes", "note": "service_rate +10%"}
    for bad, why in [
        ({"verdict": "adapts", "reason": "r", "per_cell": [good_cell]},
         "unknown verdict"),
        ({"verdict": "reads_it", "per_cell": [good_cell]}, "no reason"),
        ({"verdict": "reads_it", "reason": "r"}, "no per_cell"),
        ({"verdict": "reads_it", "reason": "r", "per_cell": []}, "empty per_cell"),
        ({"verdict": "reads_it", "reason": "r",
          "per_cell": [good_cell, good_cell]}, "per_cell longer than the cells"),
        ({"verdict": "reads_it", "reason": "r",
          "per_cell": [{"cell": 1, "moved": "sort of", "note": "n"}]},
         "unknown 'moved' value"),
        ({"verdict": "reads_it", "reason": "r",
          "per_cell": [{"cell": 1, "moved": "yes"}]}, "a note-less cell entry"),
    ]:
        try:
            PA._parse_verdict(bad, n_cells=1)
        except ExtractionError:
            _check(f"contract rejects: {why}", True)
        else:
            raise AssertionError(f"contract accepted a reply with {why}")

    v = PA._parse_verdict(
        {"verdict": " READS_IT ", "reason": "income_mean +25% in both cells",
         "evidence": "income_mean column",
         "per_cell": [{"cell": 1, "moved": "YES", "note": "assigned +10%"},
                      {"cell": 2, "moved": "no", "note": "flat"}]},
        n_cells=2)
    _check("a well-formed verdict parses and normalises its case",
           v.verdict == "reads_it" and v.status == "reads_it"
           and v.per_cell[0]["moved"] == "yes")


def check_fail_open() -> None:
    class Dead:
        def complete(self, *a, **k):
            raise RuntimeError("connection reset")

    scenes = [FakeScene(seed=11)]
    objs = [FakeObjective()]
    cells = _patched(lambda _s: PA.probe_phase2(object(), {}, scenes, objs,
                                                log=lambda _x: None))
    lines: List[str] = []
    v = PA.audit_policy(Dead(), phase=2, objective="o", description="d",
                        code="c", cells=cells, log=lines.append)
    _check("a dead audit call does not raise", v.status == "error")
    _check("the failure is logged, not swallowed",
           any("FAILED to run" in ln for ln in lines))

    v2 = PA.audit_policy(ScriptedAuditor(["not json at all"]), phase=2,
                         objective="o", description="d", code="c", cells=cells,
                         log=lambda _s: None)
    _check("an unparseable reply also fails open", v2.status == "error")

    two_readings = [{"cell": 1, "moved": "yes", "note": "n"},
                    {"cell": 2, "moved": "yes", "note": "n"}]
    v3 = PA.audit_policy(
        ScriptedAuditor([{"verdict": "reads_it", "reason": "r",
                          "per_cell": two_readings}]),
        phase=2, objective="o", description="d", code="c", cells=cells,
        log=lambda _s: None)
    _check("a reply whose per_cell count disagrees with the cells fails open",
           v3.status == "error")


def check_run_reports_and_applies_nothing() -> None:
    scenes = [FakeScene(seed=11), FakeScene(num_drivers=1200, seed=22)]
    objs = [FakeObjective(), FakeObjective(label="pay per shared seat")]
    cells = _patched(lambda _s: PA.probe_phase2(object(), {}, scenes, objs,
                                                log=lambda _x: None))
    client = ScriptedAuditor([{
        "verdict": "mixed",
        "reason": "cell 1 separates on income_mean (+25%); cell 2 is flat.",
        "evidence": "income_mean",
        "per_cell": [{"cell": 1, "moved": "yes", "note": "income_mean +25%"},
                     {"cell": 2, "moved": "no", "note": "columns match"}],
    }])
    lines: List[str] = []
    rep = PA.run_policy_audit(client, phase=2, objective="o", description="d",
                              code="c", cells=cells, log=lines.append)
    _check("the report carries the verdict and the per-cell readings",
           rep["verdict"] == "mixed" and len(rep["per_cell"]) == 2)
    _check("the report keeps the measured arms for every cell",
           rep["cells"][0]["arms"][PA.ARM_GIVEN]["assigned"] == 3300.0
           and rep["cells"][0]["arms"][PA.ARM_BLIND]["assigned"] == 3000.0)
    _check("the report keeps the rendered table for the artifact",
           "CELL 1" in rep["table"] and "CELL 2" in rep["table"])
    _check("the log says out loud that nothing was applied",
           any("ADVISORY ONLY" in ln for ln in lines))
    _check("the per-cell readings are printed",
           any("cell 1: moved=yes" in ln for ln in lines))

    empty = PA.run_policy_audit(client, phase=2, objective="o", description="d",
                                code="c", cells=[], log=lambda _s: None)
    _check("no cells means a skip, not a crash", empty["status"] == "skipped")


def main() -> None:
    check_phase2_arms_differ_in_one_thing()
    check_phase3_adds_the_fairness_arm()
    check_table()
    check_prompt_states_facts_only()
    check_contract()
    check_fail_open()
    check_run_reports_and_applies_nothing()
    print("\n[policy-audit] ALL Phase-2/3 policy soft-check offline checks passed "
          "(no API key used).")


if __name__ == "__main__":
    main()
