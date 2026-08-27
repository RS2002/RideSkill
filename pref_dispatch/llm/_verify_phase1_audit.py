"""Offline, key-free check of the Phase-1 post-search audit (v9).

Everything here runs against a scripted client and a stubbed search -- no network,
no environment, no API key. What it pins down:

* the evidence table shows the RAW fitness and the KPI columns, and never the
  group-relative advantages (showing those is the exact blindness the audit exists
  to remove);
* the output contract is enforced: an unknown verdict, a ``description_wrong``
  without replacement words, or a ``fitness_wrong`` without a complaint are all
  rejected rather than silently accepted;
* a broken audit call FAILS OPEN -- the champion survives, stamped
  ``audit.status = "error"`` -- because throwing away a finished search over a
  broken judge is worse than freezing an unjudged skill;
* ``match`` searches once; ``description_wrong`` searches once and rewrites the
  card in place (keeping the originals); ``fitness_wrong`` re-searches up to the
  cap and then freezes with ``audit.status = "unresolved"``;
* a re-authored attempt actually RECEIVES the rejected formula, the complaint and
  the measured numbers, and attempt 3 additionally receives attempt 1's formula --
  without that, attempt 3 is free to rediscover attempt 1's mistake;
* for a self-invented skill (no external brief) the intent is pinned to the FIRST
  search's own objective, so a re-author cannot redefine its way to a "match";
* ``audit_feedback`` reaches generation 0 -- the only generation that writes a
  fitness -- and no later prompt;
* the three sandbox probes catch a fitness that divides by ``assigned`` without a
  zero guard, which the old single fake dict could not.

Run: ``python -m pref_dispatch.llm._verify_phase1_audit``
"""

from __future__ import annotations

import json
import random
from typing import Dict, List, Optional

from pref_dispatch.llm import skill_audit as SA
from pref_dispatch.llm.evolve import Candidate
from pref_dispatch.llm.evolve_skill_group import SkillGroupEval
from pref_dispatch.llm.extract import ExtractionError
from pref_dispatch.llm.prompts.skill_evolve import build_skill_prompt
from pref_dispatch.llm.sandbox import compile_fitness, fitness_probes, validate_fitness

_CODE = ("def score(driver_obs, order, phi_ep, phi_step):\n"
         "    return 1.0\n\n\n"
         "def noop_score(driver_obs, phi_ep, phi_step):\n"
         "    return 0.0\n")

_FIT_BAD = ("def fitness(metrics):\n"
            "    return metrics['completed'] - 0.5 * metrics['detour_total']\n")
_FIT_BAD2 = ("def fitness(metrics):\n"
             "    return -metrics['mean_service_time']\n")
_FIT_OK = "def fitness(metrics):\n    return metrics['revenue']\n"


def _check(label: str, ok: bool, detail: str = "") -> None:
    if not ok:
        raise AssertionError(f"{label} FAILED {detail}")
    print(f"[audit] {label} OK {detail}".rstrip())


# --------------------------------------------------------------------------- #
# Fixtures.                                                                   #
# --------------------------------------------------------------------------- #
def _eval(*, served: bool) -> SkillGroupEval:
    """A two-scene record: either a working skill or one that serves nobody.

    The advantages are deliberately HEALTHY-LOOKING in both cases (+0.87 / -0.87,
    the shape the direction-2 incident actually printed) so any check that passes
    only because the advantages look bad would fail here.
    """
    if served:
        mets = [
            {"revenue": 107366.2, "service_rate": 0.982, "completed": 6225,
             "assigned": 8430, "mean_service_time": 11.76, "detour_total": 37940.6,
             "income_gini": 0.156, "income_min": -5.488},
            {"revenue": 36390.0, "service_rate": 0.205, "completed": 1179,
             "assigned": 1760, "mean_service_time": 17.52, "detour_total": 10798.6,
             "income_gini": 0.168, "income_min": -4.835},
        ]
        raw = [107366.2, 36390.0]
    else:
        zero = {"revenue": 0.0, "service_rate": 0.0, "completed": 0, "assigned": 0,
                "mean_service_time": 0.0, "detour_total": 0.0,
                "income_gini": 0.0, "income_min": 0.0}
        mets = [dict(zero), dict(zero)]
        raw = [0.0, 0.0]
    return SkillGroupEval(
        fitness=0.867, raw_fitness=0.867,
        per_scenario_adv=[0.84, 0.90],          # the misleading number
        per_scenario_raw=raw,
        per_scenario_metrics=mets,
        labels=["fleet1829_peak", "fleet200_offpeak"],
        per_band={"fleet1000-1500": 0.90, "fleet200-500": 0.84},
    )


def _cand(*, served: bool, fitness_code: str = _FIT_BAD,
          objective: str = "minimise rider waiting and in-car time") -> Candidate:
    from pref_dispatch.llm.sandbox import compile_skill
    meta = {
        "skill_name": "probe_skill",
        "objective": objective,
        "mechanism": "threshold on pickup time",
        "description": "Serves the riders whose combined wait and ride is shortest.",
        "fitness_code": fitness_code,
        "fitness_rationale": "Waiting and detour are the costs, so subtract them.",
        "code": _CODE,
        "gen": 3,
    }
    return Candidate(meta=meta, skill=compile_skill(_CODE, name="probe_skill"),
                     fitness_fn=lambda m: 0.0, evaluation=_eval(served=served))


class ScriptedAuditor:
    """Returns a canned verdict per call; records every prompt it was sent."""

    def __init__(self, verdicts: List[Dict]) -> None:
        self.verdicts = list(verdicts)
        self.prompts: List[str] = []

    def complete(self, system: str, user: str, temperature=None) -> str:
        self.prompts.append(user)
        v = self.verdicts[min(len(self.prompts) - 1, len(self.verdicts) - 1)]
        if isinstance(v, str):        # a raw (possibly malformed) reply
            return v
        return json.dumps(v)


class SearchStub:
    """Stands in for ``evolve_skill_group``; records the audit_feedback it got."""

    def __init__(self, cands: List[Candidate]) -> None:
        self.cands = list(cands)
        self.feedback: List[Optional[str]] = []

    def __call__(self, client, env_profile, *, audit_feedback=None, log=print, **kw):
        self.feedback.append(audit_feedback)
        return self.cands[min(len(self.feedback) - 1, len(self.cands) - 1)]


def _run(verdicts: List[Dict], cands: List[Candidate], *,
         intent: Optional[str] = "Minimise passenger waiting and in-car time.",
         max_reauthor: int = SA.DEFAULT_MAX_REAUTHOR):
    """Drive ``evolve_skill_audited`` with the search stubbed out."""
    stub = SearchStub(cands)
    client = ScriptedAuditor(verdicts)
    orig = SA.evolve_skill_group
    SA.evolve_skill_group = stub
    lines: List[str] = []
    try:
        champ = SA.evolve_skill_audited(
            client, "PROFILE", intent=intent, max_reauthor=max_reauthor,
            log=lines.append, scenarios=[], generations=1,
        )
    finally:
        SA.evolve_skill_group = orig
    return champ, stub, client, lines


# --------------------------------------------------------------------------- #
# Checks.                                                                     #
# --------------------------------------------------------------------------- #
def check_behaviour_table() -> None:
    tbl = SA.behaviour_table(_eval(served=False))
    _check("a zero-service scene is visible in the assigned column",
           tbl.count("|        0 |") == 2)
    _check("table shows the RAW fitness, not the group-relative advantage",
           "+0.84" not in tbl and "+0.90" not in tbl and "+0.00" in tbl)
    # v10: the table is columns only. A footer counting the zero-assignment scenes
    # is a true number, but it is a count of the ONE symptom we had already picked
    # out -- i.e. our diagnosis, handed to the model and read back as its own.
    _check("table carries no summary line singling out zero service",
           "ZERO orders" not in tbl and "never moved" not in tbl)

    tbl2 = SA.behaviour_table(_eval(served=True))
    _check("table carries the KPI columns and the scene labels",
           "fleet1829_peak" in tbl2 and "107366" in tbl2 and "0.982" in tbl2)
    _check("a working skill's assigned counts are shown as measured",
           "8430" in tbl2 and "1760" in tbl2)

    ev = _eval(served=True)
    ev.per_scenario_raw = [float("nan"), 36390.0]
    ev.per_scenario_metrics[0] = None
    _check("a crashed scene is shown as CRASHED, not as a zero row",
           "CRASHED" in SA.behaviour_table(ev))
    _check("a champion with no record says so",
           "no per-scenario record" in SA.behaviour_table(None))


def check_contract() -> None:
    for bad, why in [
        ({"verdict": "looks_fine", "reason": "r"}, "unknown verdict"),
        ({"verdict": "match"}, "no reason"),
        ({"verdict": "description_wrong", "reason": "r"}, "no new_description"),
        ({"verdict": "fitness_wrong", "reason": "r"}, "no fitness_complaint"),
    ]:
        try:
            SA._parse_verdict(bad, attempt=1)
        except ExtractionError:
            _check(f"contract rejects: {why}", True)
        else:
            raise AssertionError(f"contract accepted a reply with {why}")

    v = SA._parse_verdict(
        {"verdict": "MATCH ", "reason": "served 98% of riders", "evidence": "col 2"},
        attempt=2)
    _check("a well-formed verdict parses and normalises its case",
           v.verdict == "match" and v.status == "match" and v.attempt == 2)


def check_fail_open() -> None:
    class Dead:
        def complete(self, *a, **k):
            raise RuntimeError("connection reset")

    lines: List[str] = []
    v = SA.audit_skill(Dead(), _cand(served=True), intent="x", log=lines.append)
    _check("a dead audit call keeps the champion",
           v.verdict == "match" and v.status == "error")
    _check("the failure is logged, not swallowed",
           any("FAILED to run" in ln for ln in lines))

    v2 = SA.audit_skill(ScriptedAuditor(["not json at all"]), _cand(served=True),
                        intent="x", log=lambda _s: None)
    _check("an unparseable reply also fails open", v2.status == "error")


def check_match_path() -> None:
    champ, stub, client, _ = _run(
        [{"verdict": "match", "reason": "served 98%/20% of riders",
          "evidence": "service_rate column"}],
        [_cand(served=True)])
    _check("a match searches exactly once", len(stub.feedback) == 1)
    _check("the first search gets no audit feedback", stub.feedback == [None])
    _check("the verdict is stamped into the frozen meta",
           champ.meta["audit"]["status"] == "match"
           and champ.meta["audit"]["searches"] == 1)
    _check("the audit prompt shows the raw numbers and hides the advantages",
           "107366" in client.prompts[0] and "+0.84" not in client.prompts[0])
    _check("the audit prompt states what the skill was ASKED to be",
           "Minimise passenger waiting" in client.prompts[0])


def check_description_path() -> None:
    champ, stub, _, lines = _run(
        [{"verdict": "description_wrong",
          "reason": "it serves long fares, not short waits",
          "evidence": "revenue per order",
          "new_objective": "maximise served trip-minutes",
          "new_description": "Prefers long fares and tolerates a longer pickup."}],
        [_cand(served=True)])
    _check("a wrong description does NOT trigger a re-search", len(stub.feedback) == 1)
    _check("the card's words are replaced",
           champ.meta["description"].startswith("Prefers long fares")
           and champ.meta["objective"] == "maximise served trip-minutes")
    _check("the original words are kept next to the verdict",
           champ.meta["audit"]["original_description"].startswith("Serves the riders")
           and "waiting" in champ.meta["audit"]["original_objective"])
    _check("the rewrite is announced in the log",
           any("description REWRITTEN" in ln for ln in lines))


def check_fitness_retry_and_cap() -> None:
    bad = {"verdict": "fitness_wrong",
           "reason": "it assigned 0 orders on both scenes; do-nothing scores 0",
           "evidence": "assigned column is 0",
           "fitness_complaint": "detour_total is in tens of thousands of minutes "
                                "while completed is in thousands, so -0.5*detour "
                                "dominates and refusing every order is the maximum"}
    cands = [_cand(served=False, fitness_code=_FIT_BAD),
             _cand(served=False, fitness_code=_FIT_BAD2),
             _cand(served=False, fitness_code=_FIT_OK)]
    champ, stub, client, lines = _run([bad], cands)

    _check("a wrong fitness re-searches up to the cap", len(stub.feedback) == 3)
    _check("the first search is unprompted, the retries are not",
           stub.feedback[0] is None and all(f for f in stub.feedback[1:]))
    fb2, fb3 = stub.feedback[1], stub.feedback[2]
    _check("the retry carries the REJECTED formula", "detour_total" in fb2)
    _check("the retry carries the complaint", "dominates" in fb2)
    _check("the retry carries the measured numbers",
           "fleet1829_peak" in fb2 and "fleet200_offpeak" in fb2)
    _check("the retry restates the unchanged intent",
           "Minimise passenger waiting" in fb2)
    _check("attempt 3 also sees attempt 1's rejected formula",
           "ALSO ALREADY REJECTED" in fb3 and "detour_total" in fb3)
    _check("the audit prompt says which attempt this is",
           "audit attempt 2 of at most 3" in client.prompts[1])
    _check("hitting the cap freezes the last champion as unresolved",
           champ.meta["audit"]["status"] == "unresolved"
           and champ.meta["fitness_code"] == _FIT_OK)
    _check("the cap is announced in the log",
           any("CAP REACHED" in ln for ln in lines))

    # Second attempt passing -> stop there, no third search.
    ok = {"verdict": "match", "reason": "now serves 98%/20%", "evidence": "assigned"}
    champ2, stub2, _, _ = _run([bad, ok], cands)
    _check("a re-authored fitness that passes stops the loop",
           len(stub2.feedback) == 2 and champ2.meta["audit"]["status"] == "match"
           and champ2.meta["audit"]["searches"] == 2)

    champ3, stub3, _, _ = _run([bad], cands, max_reauthor=0)
    _check("max_reauthor=0 means one search and no retry",
           len(stub3.feedback) == 1 and champ3.meta["audit"]["status"] == "unresolved")


def check_self_invented_intent_is_pinned() -> None:
    """A QD skill is judged against its FIRST objective, not each attempt's."""
    bad = {"verdict": "fitness_wrong", "reason": "served nobody",
           "evidence": "assigned 0", "fitness_complaint": "cost term dominates"}
    cands = [_cand(served=False, objective="pack near-full cars with detour-cheap riders"),
             _cand(served=False, objective="maximise revenue"),      # drifted
             _cand(served=False, objective="maximise revenue")]
    _, stub, client, lines = _run([bad], cands, intent=None)
    _check("a self-invented skill takes its intent from gen 0",
           any("judging against its own gen-0 objective" in ln for ln in lines))
    _check("every retry is judged against that SAME intent",
           all("pack near-full cars" in p for p in client.prompts),
           f"({len(client.prompts)} audit prompt(s))")
    _check("the retry brief also restates the pinned intent",
           "pack near-full cars" in (stub.feedback[1] or ""))


def check_feedback_reaches_generation_zero_only() -> None:
    """The founder prompt carries the audit brief; the improve prompts must not."""
    p = build_skill_prompt("PROFILE", objective_hint="x",
                           audit_feedback="MARKER-AUDIT-FEEDBACK")
    _check("build_skill_prompt renders the audit brief",
           "MARKER-AUDIT-FEEDBACK" in p["user"]
           and "A PREVIOUS SEARCH UNDER THIS SAME BRIEF WAS REJECTED" in p["user"])
    _check("without it nothing is rendered",
           "REJECTED" not in build_skill_prompt("PROFILE")["user"])

    # End-to-end through the real search loop, with the rollout stubbed.
    from pref_dispatch.llm import evolve_skill_group as G
    from pref_dispatch.llm._verify_phase1_group import (
        _SCENARIOS, RolloutStub, ScriptedClient,
    )

    client = ScriptedClient()
    orig = G.rollout_skill_on_scenario
    G.rollout_skill_on_scenario = RolloutStub()
    try:
        G.evolve_skill_group(
            client, "PROFILE", objective_hint="maximise revenue",
            audit_feedback="MARKER-AUDIT-FEEDBACK",
            scenarios=_SCENARIOS, generations=1, mu=2, lam=2,
            crossover_rate=0.0, fresh_per_round=1,
            rng=random.Random(0), log=lambda _s: None,
        )
    finally:
        G.rollout_skill_on_scenario = orig
    hits = [i for i, p in enumerate(client.prompts) if "MARKER-AUDIT-FEEDBACK" in p]
    _check("only generation 0 sees the audit brief",
           hits and all(i < 3 for i in hits)
           and all("# FIXED OBJECTIVE" not in client.prompts[i] for i in hits),
           f"(prompts {hits} of {len(client.prompts)})")


def check_probes_catch_the_unguarded_divide() -> None:
    probes = fitness_probes()
    _check("three probes are run, including a served-nobody one",
           set(probes) == {"busy_hour", "scarce_hour", "served_nobody"}
           and probes["served_nobody"]["assigned"] == 0)
    _check("the busy probe carries REAL magnitudes",
           probes["busy_hour"]["detour_total"] > 30_000
           and probes["busy_hour"]["income_min"] < 0)

    unguarded = compile_fitness(
        "def fitness(metrics):\n"
        "    return metrics['revenue'] / metrics['assigned']\n")
    ok, msg = validate_fitness(unguarded)
    _check("an unguarded per-order divide is rejected by the idle probe",
           not ok and "served_nobody" in msg, f"({msg[:60]}...)")

    guarded = compile_fitness(
        "def fitness(metrics):\n"
        "    return metrics['revenue'] / max(1.0, metrics['assigned'])\n")
    ok2, _ = validate_fitness(guarded)
    _check("the guarded version passes", ok2)


def main() -> None:
    check_behaviour_table()
    check_contract()
    check_fail_open()
    check_match_path()
    check_description_path()
    check_fitness_retry_and_cap()
    check_self_invented_intent_is_pinned()
    check_feedback_reaches_generation_zero_only()
    check_probes_catch_the_unguarded_divide()
    print("\n[audit] ALL Phase-1 self-audit offline checks passed (no API key used).")


if __name__ == "__main__":
    main()
