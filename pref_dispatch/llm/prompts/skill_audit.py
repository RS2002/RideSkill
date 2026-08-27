"""Prompt for the Phase-1 post-search audit: did the skill do what it promised?

Phase 1 is the only phase where the LLM writes its OWN yardstick: generation 0
authors ``fitness(metrics)`` and every later variant of that skill is graded by
it. That is what makes the repository's niches self-invented rather than
hand-specified -- and it is also the one failure mode Phases 2 and 3 cannot have,
because their fitness is fixed by us. If the authored fitness turns out to reward
something other than what the skill was supposed to do, the search still runs to
completion and still returns a champion; it just returns the best answer to the
wrong question, wearing a description of the right one.

Two frozen skills failed exactly this way (their fitness's global optimum was to
refuse every order), and nothing in the run flagged it: the group-relative score
prints how far a variant is from the round's OTHER variants, so a field where
everyone served zero riders reports a healthy-looking spread of advantages.

So after the search, the champion is asked to mark its own homework -- with the
measured numbers in front of it, which is the part that makes the question
answerable. It is a SOFT check by design: the question is not "did it go idle"
(that is one symptom of one cause) but "is the behaviour in this table the
behaviour you said you were building". The three possible answers separate the
two very different repairs:

* the description is wrong  -> rewrite the description, keep the skill. Cheap, and
  it keeps the repository's interpretability claim honest: every card says what
  its skill actually does.
* the fitness is wrong      -> re-author the fitness and search again. Expensive,
  so it is capped.
* they match                -> freeze.

WHAT THIS PROMPT MAY AND MAY NOT SAY (v10, and the reason several paragraphs are
shorter than they used to be). The audit is only worth anything if the verdict is
the model's own reading of the table. So the prompt carries:

* MEASURED FACTS about the system -- what each column is, what happens
  mechanically after each verdict, how many attempts are left. A deployment user
  would legitimately tell an operator these, and none of them points at an answer.

* NOT our diagnosis of how a previous search went wrong. Earlier versions told the
  model that the classic failure signature is ``service_rate`` near 0 because every
  fitness term is a cost that doing nothing zeroes. That is us handing over the
  verdict and then reading it back as if the model had found it. It was removed;
  the two frozen zero-service skills stay in the docstring above, where the
  maintainers read them and the model does not.
"""

from __future__ import annotations

from typing import Dict, List, Optional

SYSTEM_PROMPT = """\
You are auditing a dispatch skill that you previously designed and evolved. You
are shown what you SAID the skill would do and what it MEASURABLY did. You are
blunt and evidence-driven: you quote the numbers that decide the verdict, you do
not defend the design, and you do not invent behaviour the table does not show.
You always answer with exactly one JSON object matching the requested schema.
"""

OUTPUT_CONTRACT = """\
Respond with ONE JSON object and nothing else. Schema:

{
  "verdict": "match" | "description_wrong" | "fitness_wrong",
  "reason":  "<2-4 sentences: what the measured behaviour actually IS, and how it
              relates to what was promised. Quote the specific numbers.>",
  "evidence":"<ONE sentence naming the columns/rows that decided it>",
  "new_objective":    "<REQUIRED iff verdict is description_wrong: one sentence,
                       the objective this skill actually pursues>",
  "new_description":  "<REQUIRED iff verdict is description_wrong: 2-4 sentences
                       describing the behaviour in the table, in the same style as
                       the original description>",
  "fitness_complaint":"<REQUIRED iff verdict is fitness_wrong: what the fitness
                       formula actually pays for, which term dominates it at the
                       magnitudes in the table, and what a skill maximising it is
                       therefore driven to do. Be concrete and arithmetic.>"
}

Choosing the verdict:

  "match"              The table is a recognisable instance of the intended
                       behaviour. It does not have to be a good skill: a weak
                       specialist that specialises in the right thing is a match,
                       and so is a trade-off the stated objective accepted.
                       MECHANICALLY: the skill and its current description are
                       frozen into the repository as they stand.

  "description_wrong"  The skill does something coherent, but not the thing the
                       description claims; the fitness rewards what the table
                       shows. MECHANICALLY: your `new_objective` /
                       `new_description` replace the old ones and the skill is
                       frozen. The search is not re-run.

  "fitness_wrong"      The fitness's own maximum is somewhere other than the
                       intended behaviour, so no amount of further search under it
                       can reach that behaviour. MECHANICALLY: the champion is
                       discarded, the fitness is re-authored from your
                       `fitness_complaint`, and the whole search is re-run. This
                       is the only verdict that costs a re-search, and re-searches
                       are capped.

Each verdict is applied automatically; nothing else reads this table.
"""


def build_skill_audit_prompt(
    *,
    intent: str,
    objective: str,
    description: str,
    mechanism: str,
    fitness_code: str,
    fitness_rationale: str,
    code: str,
    behaviour_table: str,
    attempt: int = 1,
    max_attempts: int = 3,
) -> Dict[str, str]:
    """Build the ``{"system", "user"}`` prompt auditing one evolved skill.

    ``intent`` is what the skill was ASKED to be: the researcher's direction text
    for a 1a skill, or -- for a self-invented 1b skill, which had no external
    brief -- the objective the model itself declared at generation 0. Either way
    the audit compares a stated intent against measured behaviour; only the source
    of the intent differs.
    """
    parts: List[str] = []

    parts.append(
        "# TASK\nYou designed the skill below, wrote the fitness function that "
        "graded it, and evolved it. The search has finished. Decide whether the "
        "skill it produced is the skill you set out to build."
    )
    parts.append("# WHAT THIS SKILL WAS SUPPOSED TO BE\n" + intent.strip())
    parts.append(
        "# WHAT YOU SAID YOU BUILT\n"
        f"objective:   {objective.strip()}\n"
        f"mechanism:   {mechanism.strip() or '(none recorded)'}\n"
        f"description: {description.strip()}"
    )
    parts.append(
        "# THE FITNESS YOU AUTHORED (this, and only this, is what the search "
        "maximised)\n"
        f"```python\n{fitness_code.strip()}\n```\n"
        f"your stated rationale: {fitness_rationale.strip() or '(none recorded)'}"
    )
    parts.append("# THE SCORE FUNCTION THAT WON\n" f"```python\n{code.strip()}\n```")
    parts.append(
        "# WHAT IT MEASURABLY DID (real rollouts, one row per scenario)\n"
        + behaviour_table
    )
    parts.append(
        "# HOW TO READ THE TABLE\n"
        "`fitness` is the RAW value of your own formula on that scenario: the "
        "absolute number the formula returns, not the group-relative score the "
        "search ranked variants by. The group-relative score is a within-round "
        "comparison against the other variants of the same round and is not shown "
        "here. Every other column is a measured KPI of that rollout."
    )
    if attempt > 1:
        parts.append(
            f"# NOTE\nThis is audit attempt {attempt} of at most {max_attempts}. "
            "The fitness was already re-authored after a previous audit."
        )
    parts.append("# OUTPUT FORMAT\n" + OUTPUT_CONTRACT)

    return {"system": SYSTEM_PROMPT, "user": "\n\n".join(parts)}


def founder_feedback(
    *,
    intent: str,
    bad_fitness_code: str,
    complaint: str,
    reason: str,
    behaviour_table: str,
    attempt: int,
    max_attempts: int,
    earlier: Optional[List[str]] = None,
) -> str:
    """The text handed back to generation 0 when a fitness is re-authored.

    Carries the rejected formula, the audit's arithmetic complaint and the real
    numbers it was measured on. Earlier rejected formulas are listed too: without
    them attempt 3 is free to rediscover attempt 1's mistake.
    """
    parts = [
        f"YOUR PREVIOUS FITNESS FUNCTION WAS REJECTED AFTER A FULL SEARCH "
        f"(re-author {attempt} of at most {max_attempts}).",
        "",
        "The search ran to completion and returned the best skill available under "
        "this formula. An audit of that skill's MEASURED behaviour against the "
        "intent found the formula itself is not asking for the intended thing:",
        "",
        f"```python\n{bad_fitness_code.strip()}\n```",
        "",
        f"WHAT IS WRONG WITH IT: {complaint.strip()}",
        f"WHAT THE SKILL ACTUALLY DID: {reason.strip()}",
        "",
        "The measured behaviour it produced:",
        behaviour_table,
        "",
        f"THE INTENT IS UNCHANGED: {intent.strip()}",
        "",
        "Write a DIFFERENT fitness -- not the same formula with new coefficients. "
        "Before you submit, evaluate your new formula by hand on the TYPICAL "
        "metric ranges given above and check that its maximum sits on the "
        "behaviour the intent describes. The score body may also change; the "
        "fitness is the part that must.",
    ]
    if earlier:
        parts.extend([
            "",
            "ALSO ALREADY REJECTED (do not return to these):",
            *[f"  - {c.strip()}" for c in earlier],
        ])
    return "\n".join(parts)
