"""Prompt: ask the model to INVENT new natural-language objective briefs.

Phase-2 trains a combiner to read whatever objective ``w`` arrives, so what it is
trained on has to be a genuinely wide *distribution* of objectives -- not nine
hand-written sentences recycled every round. v6 item 7 hands that job to the model:
each round it proposes fresh briefs and is explicitly told what has already been
used, so the batch keeps moving instead of orbiting the same trade-off.

The model writes ENGLISH only. The brief is then translated into a concrete
``reward(event)`` by :func:`pref_dispatch.llm.evolve_reward.author_reward` on the
usual sandbox-validated path -- this prompt never asks for and never accepts code,
so nothing here widens what gets executed.

v10: the brief must describe an ADDITIVE objective (a per-event price list) -- see
:data:`pref_dispatch.llm.prompts.common.LINEAR_OBJECTIVE_RULE`, which is pasted
into this prompt and into the reward-authoring prompt so the two halves cannot
disagree. Without it the proposer invents thresholds and ratios, and the trainer
ends up on a distribution the rest of the system no longer contains.
"""

from __future__ import annotations

from typing import Dict, Sequence

from pref_dispatch.llm.prompts.common import LINEAR_OBJECTIVE_RULE

SYSTEM = (
    "You design EVALUATION OBJECTIVES for a ride-pooling fleet. You write short "
    "English briefs describing what a city operator wants to maximise. You never "
    "write code. You answer with one JSON object and nothing else."
)

# The event fields a brief may talk about. Kept in sync with what the reward
# authoring path can actually express -- a brief about something unmeasurable
# ("rider happiness") produces a reward that quietly ignores half its own words.
_EVENT_TERMS = """
An objective can only talk about things the simulator actually records per event:

  assignment      an order was matched to a car
  revenue         the fare collected on a trip (solo ride x party size)
  dispatch_wait   how long a passenger waited on the platform BEFORE the driver
                  was sent (now - request time)
  pickup_time     how long the driver takes to travel to the pickup point
  service_time    how long a rider waited plus rode, end-to-end
  ride_detour     the extra delivery time a pooled rider absorbed, versus an empty
                  car serving only them: (pooled drop-off) - (direct pickup ride +
                  solo ride); includes the driver's detour to reach them
  completion      a drop-off actually happened (distinct from being assigned)
  pooling         how many separate parties shared the car
  empty_move      a car moved with no rider aboard
  idle            a car sat still with no assignment
  pickup_distance how far the car drove to reach the rider (a distance, distinct
                  from the time-based pickup_time)
"""

_DIVERSITY = """
DIVERSITY IS THE POINT. Each new brief must differ from every listed brief along at
least TWO of these axes:

  1. WHICH TERMS it cares about -- a brief about drop-offs and seat-sharing is a
     different objective from one about fares and pickup distance.
  2. WHICH EVENT THE PRICE HANGS ON -- paying for an order the moment it is
     accepted, paying only when the rider is actually delivered, paying per shared
     seat, and paying per minute the car is in service are four different
     objectives even when they all sound like "serve more rides".
  3. WHICH WAY THE TRADE-OFF POINTS -- serve broadly vs. earn per trip; fill seats
     vs. keep detours short; keep cars busy vs. keep them near demand.
  4. WHAT IT PENALISES, AND HOW HARD -- some objectives are defined mostly by what
     they refuse to pay for (deadheading, idling, long waits), and an operator who
     charges more for a minute of empty driving than it pays for a minute with a
     rider aboard is a different operator from one where that charge is a rounding
     error.

Do NOT produce a brief that is the previous one with different adjectives, and do
NOT produce a brief that is a weighted blend of two listed ones -- a blend of two
objectives already in the batch teaches the fleet nothing the batch did not have.

Some briefs SHOULD be awkward. An operator who pays nothing at all for accepting an
order and only pays on the drop-off, or one that charges for every idle minute so
heavily that parking a car is worse than a bad trip, is a real objective and a hard
one; a batch of comfortable "balance revenue and service" briefs is what a policy
learns to ignore ``w`` on.

THE BATCH SHOULD COVER THE METRIC SPACE, NOT ALL IN EACH BRIEF. Each brief is ONE
operator's take and may focus on a SINGLE metric family -- one that is purely about
pickup_wait (weighted by party size for multi-rider orders), one purely about
detour, one purely about empty/idle, one purely about pooling seats -- provided the
SET as a whole spans different metrics. A batch where every brief bundles
revenue+service+detour together is a batch of one objective dressed up differently.
Aim so that, across the batch, each of these appears as a first-class concern at
least once: pickup_wait (with party-size weighting), revenue/fare, detour,
completion-vs-assignment, pooling/seating, empty/idle, and throughput.

And do not let the whole run live on one metric: a long run needs repeated briefs
from DIFFERENT families over time, so the fleet learns each axis -- not just the
axis that happened to score first.
"""

TASK = """
Propose {n} NEW objective briefs.

Each brief is ONE English sentence (<= 30 words), phrased as what the operator wants
to maximise or refuses to pay for. No numbers with units, no code, no formulas --
plain intent. Another model turns each brief into a reward function afterwards, and
it can only turn your sentence into a per-event price list, so a brief that asks for
a threshold, a ratio or an escalating bonus will be silently flattened into
something you did not write.

Return exactly:

{{"briefs": ["...", "...", ...],
  "why_different": "one sentence per brief saying which axis it moves on"}}
"""


def build_objective_prompt(
    used_briefs: Sequence[str],
    n: int = 4,
    *,
    repair_feedback: str = "",
) -> Dict[str, str]:
    """Prompt for ``n`` fresh briefs that differ from everything in ``used_briefs``.

    ``used_briefs`` should be every brief the run has already trained on (this
    round's and previous rounds'), newest last -- the model is told to move away
    from ALL of them, which is what keeps a long run from cycling.
    """
    parts = [_EVENT_TERMS, "# WHAT AN OBJECTIVE MAY BE\n" + LINEAR_OBJECTIVE_RULE]
    if used_briefs:
        listed = "\n".join(f"  - {b}" for b in used_briefs[-40:])
        parts.append("# ALREADY USED (do not repeat, do not paraphrase)\n" + listed)
    else:
        parts.append("# ALREADY USED\n  (none yet -- this is the first round)")
    parts.append(_DIVERSITY)
    parts.append("# TASK\n" + TASK.format(n=int(n)))
    if repair_feedback:
        parts.append("# YOUR LAST REPLY WAS REJECTED\n" + repair_feedback +
                     "\nReturn ONLY the JSON object described above.")
    return {"system": SYSTEM, "user": "\n\n".join(parts)}
