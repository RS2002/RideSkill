"""Prompt for the Phase-2 / Phase-3 post-search soft check: does it act on ``w``?

Phase 1 audits a skill against its own authored fitness (see
:mod:`pref_dispatch.llm.prompts.skill_audit`). Phases 2 and 3 cannot have that
failure -- their fitness is fixed by us -- but they have a different one, and it is
the one the whole method claims not to have: the combiner and the repositioner are
supposed to READ the episode objective ``w`` and dispatch differently because of
it. A program that ignores ``w`` entirely still trains, still wins its rounds, and
still freezes; it is simply a good objective-blind dispatcher wearing a description
that says it adapts.

The measurement is a counterfactual, not a story. The same demand hour is rolled
twice with the same seed and the same env reward; the ONLY difference is whether
the dispatcher was handed ``w``. Whatever separates the two KPI columns is the
value of reading the objective, in units the paper already reports. Phase 3 adds a
third arm at fairness strength 0, which isolates the second axis the same way.

WHERE THE OBJECTIVES COME FROM, AND WHY NOT OURS. The cells are graded on
objectives the MODEL wrote (the ``nl`` family: it proposes an English brief, then
authors a reward from it), drawn on TRAIN-split demand windows. Auditing on the
hand-written objectives of the evaluation grid would be scoring the method on its
own test set and calling the result a diagnostic.

WHAT THIS PROMPT MAY AND MAY NOT SAY. Same rule as the Phase-1 audit. It carries
MEASURED FACTS -- how the arms differ, what each column is, what happens to the
answer -- because a deployment user would legitimately state those. It carries NO
expectation of which way any column should move under any objective: that is the
question being asked, and supplying it turns the check into us reading our own
prediction back out of the model. In particular the cells' objective text is the
authoring model's own words, printed verbatim, not our summary of what it wants.

ADVISORY ONLY, by construction. Nothing here enters any fitness, gates any freeze,
or triggers any retry (the user's standing instruction: objective-responsiveness is
handled in the evolve prompts, and blindness stays a printed diagnostic). The
verdict is stored next to the frozen artifact and read by a person.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

SYSTEM_PROMPT = """\
You are auditing a dispatch program that you previously designed and evolved. You
are shown what you SAID it would do and a set of paired rollouts that isolate one
thing: what changed when the program was handed the episode objective instead of
being run without it. You are blunt and evidence-driven: you quote the numbers that
decide the verdict, you do not defend the design, and you do not invent behaviour
the table does not show. You always answer with exactly one JSON object matching
the requested schema.
"""

#: Verdicts the reply may carry. Deliberately not ordered best-to-worst -- nothing
#: downstream acts on them, so there is no incentive to report one over another.
VERDICTS = ("reads_it", "moves_elsewhere", "no_movement", "mixed")

OUTPUT_CONTRACT = """\
Respond with ONE JSON object and nothing else. Schema:

{
  "per_cell": [
     {"cell": <1-based cell number>,
      "moved": "yes" | "no" | "unclear",
      "note": "<ONE sentence: which columns separate the arms in this cell, by how
                much, and how that relates to what THIS cell's objective pays for>"}
     ... one entry per cell, in order ...
  ],
  "verdict": "reads_it" | "moves_elsewhere" | "no_movement" | "mixed",
  "reason":  "<2-4 sentences summarising the per-cell readings. Quote specific
              numbers.>",
  "evidence":"<ONE sentence naming the cells and columns that decided it>"
}

Fill `per_cell` FIRST, one entry per cell, before you choose the verdict: the
verdict is a summary of those readings and nothing else.

The verdicts:

  "reads_it"        Across the cells, the arm that was given the objective behaves
                    differently from the arm that was not, and the differences
                    follow from what those objectives pay for.
  "moves_elsewhere" The arms do differ, but the differences do not follow from the
                    price lists shown -- the program changes behaviour on something
                    other than what the objective asks for.
  "no_movement"     The arms are the same to within rounding: handing over the
                    objective changed nothing measurable.
  "mixed"           The cells do not agree with each other. Say in `reason` which
                    cells fall which way.

MECHANICALLY: nothing is applied from this answer. The program is already frozen
and this run is finished; the verdict is recorded next to the artifact and read by
a person.
"""


def _fmt_cells(cell_blocks: Sequence[str]) -> str:
    return "\n\n".join(cell_blocks)


def build_policy_audit_prompt(
    *,
    phase: int,
    objective: str,
    description: str,
    code: str,
    arm_spec: str,
    cell_blocks: Sequence[str],
    column_glossary: str,
) -> Dict[str, str]:
    """Build the ``{"system", "user"}`` prompt auditing one frozen Phase-2/3 program.

    ``arm_spec`` states, in mechanical terms, how the arms of every cell were
    produced (built by :func:`pref_dispatch.llm.policy_audit.arm_spec`).
    ``cell_blocks`` are the rendered per-cell tables, already carrying each cell's
    scene, its objective text as authored, and its arm columns.
    """
    what = "combiner" if int(phase) == 2 else "repositioner"
    parts: List[str] = []

    parts.append(
        f"# TASK\nYou designed the {what} below and evolved it. The search has "
        "finished and it is frozen. Below are paired rollouts that isolate what "
        "changed when it was handed the episode objective. Read them and say what "
        "they show."
    )
    parts.append(
        "# WHAT YOU SAID YOU BUILT\n"
        f"objective:   {objective.strip() or '(none recorded)'}\n"
        f"description: {description.strip() or '(none recorded)'}"
    )
    parts.append(f"# THE PROGRAM THAT WON\n```python\n{code.strip()}\n```")
    parts.append("# HOW THE ARMS WERE PRODUCED\n" + arm_spec.strip())
    parts.append("# WHAT THE COLUMNS ARE\n" + column_glossary.strip())
    parts.append(
        f"# THE CELLS ({len(cell_blocks)} of them; real rollouts)\n"
        + _fmt_cells(cell_blocks)
    )
    parts.append("# OUTPUT FORMAT\n" + OUTPUT_CONTRACT)

    return {"system": SYSTEM_PROMPT, "user": "\n\n".join(parts)}
