"""Phase-1 post-search audit: does the evolved skill match the skill we asked for?

Why this module exists
----------------------
Phase 1 is the only phase whose fitness function is written by the model. Generation
0 authors ``fitness(metrics)``; every later variant of that skill is graded by it and
nothing may change it. That is deliberate -- it is what lets the repository invent its
own niches instead of us hand-specifying them -- but it means a badly-authored fitness
is unrecoverable inside the run: the search will maximise it faithfully and hand back a
champion, and that champion's card will still carry the description of the skill we
meant to build.

Two frozen skills failed exactly this way. Their formulas were

    completed + 40*service_rate - 0.5*detour_total      # about -29,600 on a real hour
    service_rate - 0.6*mean_service_time - 0.4*detour_per_order   # about -10.6

and in both, "refuse every order" scores exactly 0 -- the global maximum. The runs did
not go wrong; the yardstick did. Worse, nothing in the log said so: fitness is reported
group-relative (how far this variant is from the round's other variants), so a round in
which every variant served zero riders still prints a healthy spread of advantages. The
direction-2 record shows ``fitness +0.867`` next to ``per_scenario_raw [0,0,0,0,0,0]``.

The check
---------
After a search finishes, the champion is shown (a) what it said it would build and
(b) the RAW measured numbers it actually produced, and is asked which of three things
happened. It is a soft judgement on purpose. "Did it go idle" is one symptom of one
cause; the question that matters is whether the behaviour in the table is the behaviour
that was promised, which also catches a skill that serves 72% of orders while pursuing
something entirely different from its brief.

    match              -> freeze it.
    description_wrong  -> the fitness is fine and the skill is coherent, the words are
                          wrong. Rewrite the card's objective/description; do NOT
                          re-search. A correctly-labelled skill in a slightly different
                          niche is worth keeping, and it keeps every frozen card honest.
    fitness_wrong      -> the formula's maximum is not the intended behaviour, so more
                          search cannot help. Re-author the fitness with the rejected
                          formula, the complaint and the real numbers attached, and run
                          the search again.

Re-authoring is capped (default 2, so at most 3 searches). Past the cap the last
champion is frozen anyway and its meta is stamped ``audit.status = "unresolved"`` --
a run that has burned three full searches should not be dropped silently, and the tag
makes the doubt visible in the frozen artifact rather than only in a log.

Nothing here can fail a run. If the audit call itself dies or comes back unparseable,
the champion is kept and stamped ``audit.status = "error"``: a broken judge must not
throw away a completed search.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from pref_dispatch.llm.client import LLMClient
from pref_dispatch.llm.evolve import Candidate
from pref_dispatch.llm.evolve_skill_group import SkillGroupEval, evolve_skill_group
from pref_dispatch.llm.extract import ExtractionError, extract_json
from pref_dispatch.llm.prompts.skill_audit import (
    build_skill_audit_prompt,
    founder_feedback,
)

#: Verdicts the auditor may return (plus the two internal statuses below).
VERDICTS = ("match", "description_wrong", "fitness_wrong")

#: How many times a rejected fitness may be re-authored (searches = 1 + this).
DEFAULT_MAX_REAUTHOR = 2

#: Judgement, not generation -- keep it near-deterministic.
AUDIT_TEMPERATURE = 0.2


# --------------------------------------------------------------------------- #
# The evidence table.                                                         #
# --------------------------------------------------------------------------- #
def behaviour_table(ev: Optional[SkillGroupEval]) -> str:
    """Render one champion's measured per-scenario behaviour as a text table.

    Deliberately shows ``per_scenario_raw`` (the fitness formula's own absolute
    value) and never ``per_scenario_adv`` (the group-relative score). The whole
    failure this audit exists to catch is invisible in the advantages: a field where
    every variant served nobody still standardises to a healthy-looking spread.

    v10: the table is columns only. It used to end with a footer counting how many
    scenarios assigned zero orders -- a true count, but a count of the ONE symptom we
    had already decided was the interesting one, which is us making the diagnosis and
    then reading it back as the model's. The ``assigned`` column carries the same
    information without choosing what to look at.

    Costs nothing to build -- every number here was already stored by the search.
    """
    if not isinstance(ev, SkillGroupEval) or not ev.per_scenario_raw:
        return "(no per-scenario record was kept for this champion)"

    head = ("scenario                  |  fitness | assigned | serv_rate | completed "
            "| svc_min | detour/order | revenue | gini | income_min")
    rule = "-" * len(head)
    rows: List[str] = [head, rule]

    for i, raw in enumerate(ev.per_scenario_raw):
        label = ev.labels[i] if i < len(ev.labels) else f"scene{i}"
        met = ev.per_scenario_metrics[i] if i < len(ev.per_scenario_metrics) else None
        if met is None or not math.isfinite(float(raw)):
            rows.append(f"{label:<25} |   CRASHED (the skill raised; no episode ran)")
            continue
        assigned = float(met.get("assigned", 0.0))
        det_per = (float(met.get("detour_total", 0.0)) / assigned) if assigned > 0 else 0.0
        rows.append(
            f"{label:<25} | {float(raw):+8.2f} | {assigned:8.0f} | "
            f"{float(met.get('service_rate', 0.0)):9.3f} | "
            f"{float(met.get('completed', 0.0)):9.0f} | "
            f"{float(met.get('mean_service_time', 0.0)):7.2f} | "
            f"{det_per:12.2f} | {float(met.get('revenue', 0.0)):7.0f} | "
            f"{float(met.get('income_gini', 0.0)):4.2f} | "
            f"{float(met.get('income_min', 0.0)):10.2f}"
        )

    rows.append(rule)
    return "\n".join(rows)


# --------------------------------------------------------------------------- #
# The verdict.                                                                #
# --------------------------------------------------------------------------- #
@dataclass
class AuditVerdict:
    """One audit reply, already validated against the output contract.

    ``status`` is the operational outcome and is what gets stamped into the frozen
    skill's meta: it is the verdict for a normal answer, ``"error"`` when the audit
    call or its parse failed, and ``"unresolved"`` when the re-author cap was hit
    with the fitness still rejected.
    """

    verdict: str
    reason: str = ""
    evidence: str = ""
    new_objective: str = ""
    new_description: str = ""
    fitness_complaint: str = ""
    status: str = ""
    attempt: int = 1
    raw: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.status:
            self.status = self.verdict


def _parse_verdict(obj: Dict, *, attempt: int) -> AuditVerdict:
    """Validate an audit reply against the contract in ``prompts/skill_audit``."""
    verdict = str(obj.get("verdict", "")).strip().lower()
    if verdict not in VERDICTS:
        raise ExtractionError(
            f"verdict must be one of {VERDICTS}, got {obj.get('verdict')!r}")
    reason = str(obj.get("reason", "")).strip()
    if not reason:
        raise ExtractionError("audit reply has no 'reason'")

    new_obj = str(obj.get("new_objective", "")).strip()
    new_desc = str(obj.get("new_description", "")).strip()
    complaint = str(obj.get("fitness_complaint", "")).strip()

    if verdict == "description_wrong" and not new_desc:
        raise ExtractionError(
            "verdict 'description_wrong' requires a non-empty 'new_description' "
            "(the whole point of that verdict is to replace the card's words)")
    if verdict == "fitness_wrong" and not complaint:
        raise ExtractionError(
            "verdict 'fitness_wrong' requires a non-empty 'fitness_complaint' "
            "(the next generation 0 is given it verbatim and cannot re-author "
            "the formula without it)")

    return AuditVerdict(
        verdict=verdict,
        reason=reason,
        evidence=str(obj.get("evidence", "")).strip(),
        new_objective=new_obj,
        new_description=new_desc,
        fitness_complaint=complaint,
        attempt=attempt,
        raw=dict(obj),
    )


def audit_skill(
    client: LLMClient,
    cand: Candidate,
    *,
    intent: str,
    attempt: int = 1,
    max_attempts: int = 1 + DEFAULT_MAX_REAUTHOR,
    table: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> AuditVerdict:
    """Ask the model whether ``cand`` is the skill ``intent`` asked for.

    Never raises. A failed call or an unparseable reply comes back as
    ``AuditVerdict(verdict="match", status="error")`` -- fail-open, because throwing
    away a completed search over a broken judge is strictly worse than freezing a
    skill whose audit we could not run. The ``"error"`` status is stamped into the
    artifact either way, so it is visible afterwards.
    """
    tbl = table if table is not None else behaviour_table(
        cand.evaluation if isinstance(cand.evaluation, SkillGroupEval) else None)
    prompt = build_skill_audit_prompt(
        intent=intent,
        objective=str(cand.meta.get("objective", "")),
        description=str(cand.meta.get("description", "")),
        mechanism=str(cand.meta.get("mechanism", "")),
        fitness_code=str(cand.meta.get("fitness_code", "")),
        fitness_rationale=str(cand.meta.get("fitness_rationale", "")),
        code=str(cand.meta.get("code", "")),
        behaviour_table=tbl,
        attempt=attempt,
        max_attempts=max_attempts,
    )
    try:
        reply = client.complete(prompt["system"], prompt["user"],
                                temperature=AUDIT_TEMPERATURE)
        return _parse_verdict(extract_json(reply), attempt=attempt)
    except Exception as e:  # noqa: BLE001 -- a broken judge must not kill a search
        log(f"    [audit] FAILED to run ({type(e).__name__}: {e}); "
            f"keeping the champion and stamping audit.status='error'")
        return AuditVerdict(verdict="match", status="error", attempt=attempt,
                            reason=f"audit did not run: {type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
# The wrapper the two Phase-1 call sites use.                                 #
# --------------------------------------------------------------------------- #
def _stamp(cand: Candidate, v: AuditVerdict, *, table: str, searches: int) -> None:
    """Record the audit inside the champion's meta so it freezes with the skill."""
    cand.meta["audit"] = {
        "status": v.status,
        "verdict": v.verdict,
        "reason": v.reason,
        "evidence": v.evidence,
        "fitness_complaint": v.fitness_complaint,
        "attempt": v.attempt,
        "searches": searches,
        "behaviour_table": table,
    }


def _apply_description_fix(cand: Candidate, v: AuditVerdict,
                           log: Callable[[str], None]) -> None:
    """Rewrite the card's words, keeping the originals next to the verdict."""
    audit = cand.meta.setdefault("audit", {})
    audit["original_objective"] = str(cand.meta.get("objective", ""))
    audit["original_description"] = str(cand.meta.get("description", ""))
    if v.new_objective:
        cand.meta["objective"] = v.new_objective
    cand.meta["description"] = v.new_description
    log(f"    [audit] description REWRITTEN (no re-search):")
    log(f"            objective:   {cand.meta.get('objective', '')}")
    log(f"            description: {cand.meta['description']}")


def evolve_skill_audited(
    client: LLMClient,
    env_profile: str,
    *,
    intent: Optional[str] = None,
    max_reauthor: int = DEFAULT_MAX_REAUTHOR,
    audit: bool = True,
    log: Callable[[str], None] = print,
    **kwargs,
) -> Candidate:
    """:func:`evolve_skill_group` plus the post-search audit and its retry loop.

    ``intent`` is what the skill was ASKED to be, and it is what the champion is
    judged against. For a directed (1a) skill that is the researcher's direction
    text. For a self-invented (1b/QD) skill there is no external brief, so the intent
    is taken from the FIRST search's own declared objective and then held fixed --
    otherwise a re-author could quietly redefine the target it is being graded on and
    every attempt would trivially "match".

    ``**kwargs`` are forwarded verbatim to :func:`evolve_skill_group`, so this is a
    drop-in replacement at both Phase-1 call sites. ``audit=False`` makes it exactly
    :func:`evolve_skill_group` (one search, no judging) for offline/smoke paths.

    Note on checkpoints: a re-authored attempt writes to the same checkpoint stem, so
    the resumable leader is always the LATEST attempt, while ``history.jsonl`` keeps
    every attempt -- rejected searches stay auditable without being resumable.
    """
    max_attempts = 1 + max(0, int(max_reauthor))
    feedback: Optional[str] = None
    rejected: List[str] = []
    last: Optional[Candidate] = None
    last_table = ""

    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            log(f"[audit] RE-AUTHORING the fitness and searching again "
                f"(attempt {attempt} of {max_attempts})")
        cand = evolve_skill_group(client, env_profile,
                                  audit_feedback=feedback, log=log, **kwargs)
        last = cand
        if not audit:
            return cand

        this_intent = intent or str(cand.meta.get("objective", "")).strip()
        if intent is None and attempt == 1:
            log(f"    [audit] no external brief (self-invented skill); judging "
                f"against its own gen-0 objective: {this_intent}")
            intent = this_intent

        table = behaviour_table(
            cand.evaluation if isinstance(cand.evaluation, SkillGroupEval) else None)
        last_table = table
        log("    [audit] measured behaviour of the champion:")
        for line in table.splitlines():
            log("            " + line)

        v = audit_skill(client, cand, intent=this_intent, attempt=attempt,
                        max_attempts=max_attempts, table=table, log=log)
        log(f"    [audit] verdict: {v.verdict.upper()} -- {v.reason}")
        if v.evidence:
            log(f"    [audit] evidence: {v.evidence}")
        _stamp(cand, v, table=table, searches=attempt)

        if v.verdict == "match":
            return cand
        if v.verdict == "description_wrong":
            _apply_description_fix(cand, v, log)
            return cand

        # fitness_wrong.
        log(f"    [audit] the FITNESS is the problem: {v.fitness_complaint}")
        if attempt >= max_attempts:
            break
        feedback = founder_feedback(
            intent=this_intent,
            bad_fitness_code=str(cand.meta.get("fitness_code", "")),
            complaint=v.fitness_complaint,
            reason=v.reason,
            behaviour_table=table,
            attempt=attempt,
            max_attempts=max_attempts,
            earlier=list(rejected),
        )
        rejected.append(str(cand.meta.get("fitness_code", "")))

    assert last is not None  # the loop always runs at least once
    audit_meta = last.meta.setdefault("audit", {})
    audit_meta["status"] = "unresolved"
    audit_meta["searches"] = max_attempts
    audit_meta.setdefault("behaviour_table", last_table)
    log(f"[audit] CAP REACHED after {max_attempts} search(es); freezing the last "
        f"champion with audit.status='unresolved' -- its fitness was still judged "
        f"wrong, so treat this skill's card with suspicion.")
    return last
