"""Phase-2 / Phase-3 post-search soft check: measure what reading ``w`` bought.

What this measures
------------------
Phase 1's audit asks whether an evolved skill matches its own authored fitness.
Phases 2 and 3 cannot fail that way -- their fitness is ours and is fixed -- but
they can fail the thing the method is *named* for. The combiner and the
repositioner are handed the episode objective ``w`` and are supposed to dispatch
differently because of it. A program that never touches ``w`` still trains, still
wins its rounds against the other candidates, and still freezes; it is a decent
objective-blind dispatcher carrying a description that says it adapts.

So the check is a counterfactual, not an opinion. Each cell rolls the SAME demand
hour, the SAME env reward and the SAME seed twice:

    arm "w hidden"   rollout(..., reward_fn=None)   -- the default-blind arm
    arm "w given"    rollout(..., reward_fn=w)

Everything else is byte-identical, including the reward the episode is GRADED by
(``build_env(sc, reward_function=obj.reward_function)`` in both arms). The only
difference is whether the dispatcher was allowed to look at the objective. What
separates the two KPI columns therefore IS the value of reading it, in the units
the paper already reports. Phase 3 adds a third arm at fairness strength 0 against
the same ``w``-given program, which isolates the second axis the same way.

The objectives are the MODEL's own
----------------------------------
Cells are graded on objectives drawn from the ``nl`` family -- the model proposes
an English brief, then authors a reward function from it -- on TRAIN-split demand
windows. Using the hand-written objectives of the evaluation grid would be running
the diagnostic on the test set. See
:class:`pref_dispatch.llm.objective_sampler.ObjectiveSampler`.

Advisory only
-------------
Nothing here enters any fitness, gates any freeze, or triggers any retry. The
standing instruction on this project is that objective-responsiveness is handled
in the evolve prompts, and that blindness stays a printed diagnostic. This module
prints its table, asks the model to read it, and stamps the answer next to the
frozen artifact. Like the Phase-1 audit it FAILS OPEN: a dead or unparseable audit
call is recorded as ``status="error"`` and changes nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from pref_dispatch.evaluate import DispatchController, rollout
from pref_dispatch.llm.extract import ExtractionError, extract_json
from pref_dispatch.llm.objective_sampler import _event_w
from pref_dispatch.llm.prompts.policy_audit import (
    VERDICTS,
    build_policy_audit_prompt,
)
from pref_dispatch.llm.reposition_eval import with_fairness
from pref_dispatch.reposition import Repositioner
from pref_dispatch.scenario import build_env

#: Judgement, not generation -- keep it near-deterministic (same as Phase 1).
AUDIT_TEMPERATURE = 0.2

#: How many (scene, objective) cells the probe rolls by default. Each cell costs
#: 2 full-hour rollouts in Phase 2 and 3 in Phase 3, so this is deliberately small:
#: it runs once, after the search is over, and nothing downstream waits on it.
DEFAULT_CELLS = 4

#: Arm names. These are labels for columns, not claims about the arms.
ARM_BLIND = "w hidden"
ARM_GIVEN = "w given"
ARM_FAIR0 = "w given, fairness 0"


# --------------------------------------------------------------------------- #
# Columns.                                                                    #
# --------------------------------------------------------------------------- #
#: The KPI columns shown, in order, each with the MEASURED description of what it
#: is. No column carries a preferred direction: which way a number should move
#: under a given objective is exactly the question the audit asks.
COLUMNS: Sequence[tuple] = (
    ("income_mean", "mean per-driver cumulative reward under THIS cell's reward "
                    "function -- the quantity the search itself maximised"),
    ("assigned", "orders assigned to a driver"),
    ("completed", "orders actually delivered"),
    ("service_rate", "assigned / total orders, in [0,1]"),
    ("mean_service_time", "mean end-to-end service time, minutes"),
    ("detour_total", "total extra detour time from pooling, minutes"),
    ("revenue", "sum over assigned orders of solo trip minutes x party size"),
    ("income_gini", "driver-income inequality over cumulative reward, [0,1]"),
    ("wage_gini", "driver take-home-fare inequality, [0,1]"),
    ("relocation_moves", "number of empty repositioning moves the fleet made"),
    ("reposition_distance_ratio", "share of driving distance spent repositioning "
                                  "empty"),
)

#: Columns with fewer decimals than the default, because they are counts.
_INT_KEYS = frozenset({"assigned", "completed", "relocation_moves"})


def column_glossary(keys: Sequence[str]) -> str:
    """The measured meaning of each column that survived into the table."""
    seen = {k for k in keys}
    return "\n".join(f"  {k:<26} {desc}" for k, desc in COLUMNS if k in seen)


# --------------------------------------------------------------------------- #
# The probe result.                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class Arm:
    """One rollout of one cell, named by what was withheld from the dispatcher."""

    name: str
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ProbeCell:
    """One (scene, objective) cell and its arms, all rolled on the same seed."""

    scene: str                       # the scenario, in words
    objective_text: str              # the objective AS AUTHORED, verbatim
    arms: List[Arm] = field(default_factory=list)
    label: str = ""                  # the objective's own short brief
    meta: Dict = field(default_factory=dict)


def _fmt(key: str, val: float) -> str:
    v = float(val)
    if not math.isfinite(v):
        return "n/a"
    if key in _INT_KEYS:
        return f"{v:,.0f}"
    if abs(v) >= 1000.0:
        return f"{v:,.0f}"
    return f"{v:.4g}"


def _pct(base: float, other: float) -> str:
    """Percent change of ``other`` against ``base``; ``n/a`` when base is ~0."""
    b, o = float(base), float(other)
    if not (math.isfinite(b) and math.isfinite(o)) or abs(b) < 1e-9:
        return "n/a"
    return f"{100.0 * (o - b) / abs(b):+.1f}%"


def _live_keys(cells: Sequence[ProbeCell]) -> List[str]:
    """Columns that are not identically zero across every arm of every cell.

    Drops the reposition columns from a Phase-2 table and ``wage_gini`` from a run
    with the fairness axis switched off -- all-zero columns are noise, not signal.
    """
    keys: List[str] = []
    for key, _desc in COLUMNS:
        for cell in cells:
            for arm in cell.arms:
                v = float(arm.metrics.get(key, 0.0) or 0.0)
                if math.isfinite(v) and abs(v) > 1e-12:
                    keys.append(key)
                    break
            else:
                continue
            break
    return keys


def cell_block(cell: ProbeCell, index: int, keys: Sequence[str]) -> str:
    """Render one cell: the scene, the objective as authored, then arms x KPIs."""
    lines: List[str] = [f"## CELL {index}  --  {cell.scene}"]
    lines.append("objective for this cell, as it was authored:")
    for ln in (cell.objective_text or "(none recorded)").strip().splitlines():
        lines.append("    " + ln.rstrip())

    arms = cell.arms
    widths = [max(11, len(a.name)) for a in arms]
    head = f"{'metric':<26}"
    for a, w in zip(arms, widths):
        head += f" | {a.name:>{w}}"
    for a in arms[1:]:
        head += f" | {('vs ' + arms[0].name):>{max(12, len(a.name) + 3)}}"
    lines.append("")
    lines.append(head)
    lines.append("-" * len(head))

    for key in keys:
        row = f"{key:<26}"
        base = float(arms[0].metrics.get(key, 0.0) or 0.0) if arms else 0.0
        for a, w in zip(arms, widths):
            row += f" | {_fmt(key, a.metrics.get(key, 0.0) or 0.0):>{w}}"
        for a in arms[1:]:
            wid = max(12, len(a.name) + 3)
            row += f" | {_pct(base, a.metrics.get(key, 0.0) or 0.0):>{wid}}"
        lines.append(row)
    return "\n".join(lines)


def contrast_table(cells: Sequence[ProbeCell]) -> List[str]:
    """The rendered per-cell blocks, in order (one string per cell)."""
    keys = _live_keys(cells)
    return [cell_block(c, i + 1, keys) for i, c in enumerate(cells)]


def arm_spec(phase: int) -> str:
    """State, mechanically, how the arms of every cell were produced.

    Facts only: what was held identical, what differed, and where the numbers came
    from. Nothing about what any of it is expected to show.
    """
    common = (
        f"Every cell below is ONE hour of real recorded demand. Its arms were "
        f"rolled on the SAME hour, the SAME random seed, the SAME fleet, and the "
        f"SAME reward function -- the episode is graded by that cell's objective in "
        f"every arm, so the numbers are directly comparable across a row.\n"
        f"The arms differ in exactly one thing each:\n"
        f"  '{ARM_BLIND}' -- the dispatch program was NOT given the objective. It "
        f"ran with w unavailable; every read of w returns nothing.\n"
        f"  '{ARM_GIVEN}' -- the dispatch program WAS given that cell's objective "
        f"as w, the same callable the episode is graded by. It could call it on any "
        f"event it liked.\n"
    )
    if int(phase) == 3:
        common += (
            f"  '{ARM_FAIR0}' -- as '{ARM_GIVEN}', but the fairness strength on the "
            f"episode was set to 0 instead of the value drawn for that cell. The "
            f"drawn value is printed in the scene line.\n"
        )
    common += (
        "The 'vs' columns are the percent change of that arm against the first, "
        "computed from the same numbers printed to their left."
    )
    return common


# --------------------------------------------------------------------------- #
# Rolling the arms.                                                           #
# --------------------------------------------------------------------------- #
def _scene_line(sc, *, strength: Optional[float] = None) -> str:
    dow, hr = sc.clock
    when = f"{dow} {hr:02d}:00" if dow else str(sc.regime)
    parts = [
        f"{int(sc.num_drivers)} cars",
        f"capacity {int(sc.driver_capacity)}",
        f"{float(sc.speed_kmh):g} km/h",
        f"hour {when}",
        f"seed {int(sc.seed)}",
    ]
    if strength is not None:
        parts.append(f"fairness strength {float(strength):g}")
    return ", ".join(parts)


def _objective_text(obj) -> str:
    """The objective in the authoring model's own words, plus its reward spec."""
    label = str(getattr(obj, "label", "") or "").strip()
    spec = str(getattr(obj, "spec_text", "") or "").strip()
    if label and spec:
        return f"{label}\n{spec}"
    return label or spec or "(no objective text was recorded)"


def probe_phase2(
    combiner,
    skills: Dict,
    scenarios: Sequence,
    objectives: Sequence,
    *,
    log: Callable[[str], None] = print,
) -> List[ProbeCell]:
    """Roll every (scene, objective) pair twice: w hidden, then w given.

    ``scenarios`` and ``objectives`` are zipped one-to-one, exactly as in
    :func:`pref_dispatch.llm.combiner_eval.evaluate_combiner_objectives`, and each
    arm rebuilds its env from the same :class:`~pref_dispatch.scenario.Scenario`
    so the two rollouts see byte-identical orders.
    """
    cells: List[ProbeCell] = []
    for i, (sc, obj) in enumerate(zip(scenarios, objectives)):
        reward_function = getattr(obj, "reward_function", None)
        w = _event_w(reward_function)
        arms: List[Arm] = []
        for name, use_w in ((ARM_BLIND, False), (ARM_GIVEN, True)):
            env = build_env(sc, reward_function=reward_function)
            ctrl = DispatchController(combiner, skills=skills)
            met = rollout(env, ctrl, sc.preference, seed=sc.seed,
                          reward_fn=(w if use_w else None))
            arms.append(Arm(name=name, metrics=dict(met)))
        log(f"    [policy-audit] cell {i + 1}/{len(scenarios)} rolled "
            f"({sc.label()})")
        cells.append(ProbeCell(
            scene=_scene_line(sc),
            objective_text=_objective_text(obj),
            arms=arms,
            label=str(getattr(obj, "label", "") or ""),
            meta={"scenario": sc.label(),
                  "family": str(getattr(obj, "family", ""))},
        ))
    return cells


def probe_phase3(
    scorer,
    combiner,
    skills: Dict,
    scenarios: Sequence,
    objectives: Sequence,
    strengths: Sequence[float],
    *,
    log: Callable[[str], None] = print,
) -> List[ProbeCell]:
    """Three arms per cell: w hidden, w given, and w given with fairness off.

    The first two isolate the objective axis exactly as in :func:`probe_phase2`;
    the third re-rolls the ``w``-given program with the episode's fairness strength
    set to 0 instead of the value drawn for that cell, which isolates the axis
    Phase 3 adds. Every arm carries the same frozen Phase-2 combiner and the same
    candidate scorer.
    """
    cells: List[ProbeCell] = []
    n = len(scenarios)
    for i, (sc, obj, strength) in enumerate(zip(scenarios, objectives, strengths)):
        reward_function = getattr(obj, "reward_function", None)
        w = _event_w(reward_function)
        plan = (
            (ARM_BLIND, False, float(strength)),
            (ARM_GIVEN, True, float(strength)),
            (ARM_FAIR0, True, 0.0),
        )
        arms: List[Arm] = []
        for name, use_w, stren in plan:
            env = build_env(sc, reward_function=reward_function)
            ctrl = DispatchController(
                combiner, skills=skills,
                repositioner=Repositioner(strength=1.0, scores_fn=scorer),
            )
            met = rollout(env, ctrl, with_fairness(sc.preference, stren),
                          seed=sc.seed, reward_fn=(w if use_w else None))
            arms.append(Arm(name=name, metrics=dict(met)))
        log(f"    [policy-audit] cell {i + 1}/{n} rolled "
            f"({sc.label()}, fairness {float(strength):g})")
        cells.append(ProbeCell(
            scene=_scene_line(sc, strength=float(strength)),
            objective_text=_objective_text(obj),
            arms=arms,
            label=str(getattr(obj, "label", "") or ""),
            meta={"scenario": sc.label(),
                  "family": str(getattr(obj, "family", "")),
                  "strength": float(strength)},
        ))
    return cells


# --------------------------------------------------------------------------- #
# The verdict.                                                                #
# --------------------------------------------------------------------------- #
@dataclass
class PolicyAuditVerdict:
    """One audit reply, already validated against the output contract.

    ``status`` is the verdict for a normal answer and ``"error"`` when the call or
    its parse failed; it is what gets stamped into the frozen artifact.
    """

    verdict: str
    reason: str = ""
    evidence: str = ""
    per_cell: List[Dict] = field(default_factory=list)
    status: str = ""
    raw: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.status:
            self.status = self.verdict


_MOVED = ("yes", "no", "unclear")


def _parse_verdict(obj: Dict, *, n_cells: int) -> PolicyAuditVerdict:
    """Validate an audit reply against the contract in ``prompts/policy_audit``."""
    verdict = str(obj.get("verdict", "")).strip().lower()
    if verdict not in VERDICTS:
        raise ExtractionError(
            f"verdict must be one of {VERDICTS}, got {obj.get('verdict')!r}")
    reason = str(obj.get("reason", "")).strip()
    if not reason:
        raise ExtractionError("audit reply has no 'reason'")

    raw_cells = obj.get("per_cell")
    if not isinstance(raw_cells, list) or not raw_cells:
        raise ExtractionError(
            "audit reply has no 'per_cell' list (the verdict is a summary of the "
            "per-cell readings, so those must be filled in first)")
    if len(raw_cells) != n_cells:
        raise ExtractionError(
            f"'per_cell' has {len(raw_cells)} entries for {n_cells} cells; "
            f"there must be exactly one entry per cell")

    per_cell: List[Dict] = []
    for i, c in enumerate(raw_cells, start=1):
        if not isinstance(c, dict):
            raise ExtractionError(f"per_cell[{i - 1}] is not an object")
        moved = str(c.get("moved", "")).strip().lower()
        if moved not in _MOVED:
            raise ExtractionError(
                f"per_cell entry for cell {c.get('cell', i)} has moved="
                f"{c.get('moved')!r}; must be one of {_MOVED}")
        note = str(c.get("note", "")).strip()
        if not note:
            raise ExtractionError(
                f"per_cell entry for cell {c.get('cell', i)} has no 'note'")
        per_cell.append({"cell": int(c.get("cell", i) or i),
                         "moved": moved, "note": note})

    return PolicyAuditVerdict(
        verdict=verdict,
        reason=reason,
        evidence=str(obj.get("evidence", "")).strip(),
        per_cell=per_cell,
        raw=dict(obj),
    )


def audit_policy(
    client,
    *,
    phase: int,
    objective: str,
    description: str,
    code: str,
    cells: Sequence[ProbeCell],
    log: Callable[[str], None] = print,
) -> PolicyAuditVerdict:
    """Ask the model to read the contrast table it is shown.

    Never raises. A failed call or an unparseable reply comes back as
    ``PolicyAuditVerdict(verdict="mixed", status="error")`` -- the champion is
    already frozen and nothing downstream acts on this, so a broken judge must
    cost nothing but the printed line.
    """
    blocks = contrast_table(cells)
    prompt = build_policy_audit_prompt(
        phase=int(phase),
        objective=objective,
        description=description,
        code=code,
        arm_spec=arm_spec(int(phase)),
        cell_blocks=blocks,
        column_glossary=column_glossary(_live_keys(cells)),
    )
    try:
        reply = client.complete(prompt["system"], prompt["user"],
                                temperature=AUDIT_TEMPERATURE)
        return _parse_verdict(extract_json(reply), n_cells=len(cells))
    except Exception as e:  # noqa: BLE001 -- advisory only; never kill a run
        log(f"    [policy-audit] FAILED to run ({type(e).__name__}: {e}); "
            f"recording status='error'. The champion is unaffected.")
        return PolicyAuditVerdict(
            verdict="mixed", status="error",
            reason=f"policy audit did not run: {type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
# What the two drivers call.                                                  #
# --------------------------------------------------------------------------- #
def report(cells: Sequence[ProbeCell], v: PolicyAuditVerdict) -> Dict:
    """The record stamped next to the frozen artifact (JSON-safe)."""
    return {
        "status": v.status,
        "verdict": v.verdict,
        "reason": v.reason,
        "evidence": v.evidence,
        "per_cell": list(v.per_cell),
        "cells": [
            {"scene": c.scene, "objective": c.label, **c.meta,
             "arms": {a.name: {k: float(a.metrics.get(k, 0.0) or 0.0)
                               for k, _d in COLUMNS}
                      for a in c.arms}}
            for c in cells
        ],
        "table": "\n\n".join(contrast_table(cells)),
    }


def run_policy_audit(
    client,
    *,
    phase: int,
    objective: str,
    description: str,
    code: str,
    cells: Sequence[ProbeCell],
    log: Callable[[str], None] = print,
) -> Dict:
    """Print the contrast table, ask for a reading, and return the stamp.

    Advisory end-to-end: the caller stores the returned dict on the frozen
    artifact's meta and does nothing else with it.
    """
    # The model's free text can carry characters the console encoding cannot
    # write (a Windows GBK console dies on U+2212 MINUS SIGN with a
    # UnicodeEncodeError, killing the whole run at its very last step -- it
    # happened on 2026-08-13). The audit is advisory, so its logging must never
    # be able to take the process down: replace anything unencodable.
    raw_log = log

    def log(line: str) -> None:  # noqa: A001 -- deliberate shadow, safe wrapper
        try:
            raw_log(line)
        except UnicodeEncodeError:
            import sys
            enc = getattr(sys.stdout, "encoding", None) or "utf-8"
            raw_log(line.encode(enc, errors="replace").decode(enc))
    if not cells:
        log("    [policy-audit] no cells were rolled; skipping")
        return {"status": "skipped", "verdict": "", "reason": "no cells"}

    log(f"    [policy-audit] contrast over {len(cells)} cell(s) "
        f"(same hour, same seed, same reward; w withheld vs given):")
    for block in contrast_table(cells):
        for line in block.splitlines():
            log("            " + line)

    v = audit_policy(client, phase=phase, objective=objective,
                     description=description, code=code, cells=cells, log=log)
    log(f"    [policy-audit] verdict: {v.verdict.upper()} -- {v.reason}")
    for c in v.per_cell:
        log(f"            cell {c['cell']}: moved={c['moved']} -- {c['note']}")
    if v.evidence:
        log(f"    [policy-audit] evidence: {v.evidence}")
    log("    [policy-audit] ADVISORY ONLY: nothing was retried, regraded or "
        "re-frozen because of this.")
    return report(cells, v)
