"""Prompt builder for Phase-1 skill discovery (QD + human seeds, §4).

One call asks the model to invent (or mutate) a single lower-layer scoring skill:
it names the objective, writes a cheap self-authored ``fitness(metrics)`` under
which that skill is optimised, writes the ``score`` function, and -- centrally --
explains all of it in natural language (interpretability is a headline claim).

Diversity (§4.7): the prompt shows *cards* of the skills already in the basis and
asks the model to cover a genuinely new behavioural niche, not a near-duplicate.

Robustness (§2.3): if a previous attempt failed to compile/validate, its error is
fed back so the model can repair it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from pref_dispatch.llm.prompts.common import (
    INTERPRETABILITY_RULE,
    METRIC_MENU,
    SANDBOX_RULES,
    SIGNATURE_SPEC,
    few_shot_seeds,
)

# The output contract. Explanation fields are FIRST and called out as required.
OUTPUT_CONTRACT = """\
Respond with ONE JSON object and nothing else (no prose before/after, no markdown
outside the JSON). Schema:

{
  "skill_name": "<short snake_case id, e.g. long_fare_hunter>",
  "objective":  "<ONE sentence: the single objective this skill specialises in>",
  "objective_self_check": "<2-3 sentences: WHICH objective axis this skill covers and
                 whether that axis is already covered by the existing skills. Name
                 the axis (see the OBJECTIVE AXES list below); say which listed skill
                 (if any) is close, and what is actually NOT yet covered that this
                 skill is covering. If the axis IS already covered, say so and
                 justify why a second specialist on it is still worth a skill slot.>",
  "mechanism":  "<ONE phrase naming the DECISION RULE SHAPE you used, e.g.
                 'hard feasibility gate then fare-per-minute ranking',
                 'idle-time-triggered threshold switch', 'ratio of marginal fare to
                 marginal detour', 'two-stage: shortlist by pickup, break ties by
                 pooling slack'>",
  "differs_from":"<1-2 sentences: which listed skill/mechanism yours is closest to
                 and what it does DIFFERENTLY at decision time -- not 'different
                 weights', an actually different rule>",
  "description":"<2-4 sentences: HOW the score logic realises that objective and
                 when it prefers to wait; explain behaviour, not code>",
  "fitness_code": "def fitness(metrics):\\n    # cheap scalar over the metrics dict\\n    return ...",
  "fitness_rationale": "<1-2 sentences: why this fitness measures the objective>",
  "code": "def score(driver_obs, order, phi_ep, phi_step):\\n    ...\\n\\ndef noop_score(driver_obs, phi_ep, phi_step):\\n    ..."
}

Rules for the fitness function:
  - It is YOUR self-authored reward for THIS skill; it need not be comparable to
    any other skill's fitness. It only has to rank this skill's own variants.
  - It must be a cheap pure function of the metrics dict only (arithmetic over the
    keys listed above). No rollouts, no env, no randomness, no LLM calls.
  - Same sandbox rules as the skill code (no imports; math/np only).
"""

# The MECHANISM menu shown when a proposal is asked to explore rule SHAPES rather
# than reweight one. The repository's diversity comes from behaviour, and behaviour
# comes from the shape of the decision rule -- eight skills that are all "weighted
# sum of the same four terms" differ only in coefficients and collapse onto one
# behavioural point no matter how their objectives are worded.
MECHANISM_MENU = """\
The `mechanism` field must name the SHAPE of the decision rule, and the shapes
below are all legitimate and behaviourally distinct. Pick one that the existing
repository does not already use, or invent another:

  - weighted sum of terms (the default -- already well covered, prefer something else)
  - ratio / efficiency (value per unit of a cost: fare per minute, revenue per km)
  - hard gate then rank (reject anything failing a condition, rank only survivors)
  - threshold switch on a state variable (behave one way when idle_min > T or
    demand_pressure > P, a different way otherwise)
  - two-stage / lexicographic (shortlist by one criterion, break ties by another)
  - marginal / counterfactual (score the CHANGE this order causes: added detour for
    the passengers already on board, capacity consumed, time-window slack burnt)
  - opportunity cost (what accepting this order costs you in orders you can no
    longer reach -- compare against noop_score deliberately)
  - patience / option value (a high noop_score that makes waiting a real choice, so
    the driver holds out for a better match instead of taking the first feasible one)
  - non-linear saturation (diminishing returns above a value, cliff below a value)

A skill whose mechanism is "weighted sum" with new coefficients is NOT a new skill.
"""


SYSTEM_PROMPT = """\
You are an expert in ride-POOLING dispatch and reward design. You write small,
robust, interpretable Python scoring functions for a fleet dispatcher. You reason
carefully about the objective, then produce clean code AND clear natural-language
explanations of the dispatch behaviour. You always answer with exactly one JSON
object matching the requested schema.
"""


def _skill_card(meta: Dict) -> str:
    """One-line-ish card summarising an existing basis skill (for diversity).

    Includes the skill's ``mechanism`` when it recorded one, so "do not duplicate"
    is a statement about decision RULES and not only about objective wording -- two
    skills can describe opposite objectives and still be the same weighted sum.
    """
    name = meta.get("skill_name", meta.get("name", "?"))
    obj = meta.get("objective", "(no objective recorded)")
    desc = meta.get("description", "")
    mech = str(meta.get("mechanism", "")).strip()
    head = f"- {name}: {obj}"
    if mech:
        head += f"\n    mechanism: {mech}"
    return f"{head}\n    {desc}".rstrip()



def build_skill_prompt(
    env_profile: str,
    *,
    objective_hint: Optional[str] = None,
    existing_skills: Optional[Sequence[Dict]] = None,
    similarity_note: Optional[str] = None,
    repository_note: Optional[str] = None,
    repair_feedback: Optional[str] = None,
    audit_feedback: Optional[str] = None,
) -> Dict[str, str]:
    """Build the ``{"system", "user"}`` prompt for evolving one skill.

    Parameters
    ----------
    env_profile :
        Output of :func:`pref_dispatch.llm.encode.encode_env_profile`.
    objective_hint :
        Optional steer toward a named objective (e.g. a seed extreme like
        "maximise revenue"). When ``None`` the model self-selects an uncovered
        niche -- the open-ended QD case.
    existing_skills :
        Cards (dicts with skill_name/objective/description) of skills already in
        the basis; the model is told to cover a NEW behaviour, not duplicate them.
    similarity_note :
        Always-on diversity signal (§4.7 v2): a short natural-language summary of
        how behaviourally close the last accepted/proposed skill drifted to the
        existing basis, shown EVERY round (not only after a rejection) so the model
        is proactively pushed toward an uncovered behavioural niche.
    repository_note :
        Repository state (§4.7 v3): every member with its measured redundancy, and
        -- once the repository is at its cap -- the replacement rule the proposal
        must win (beat the most-redundant incumbent or be discarded). Shown every
        round so exploration is aimed at the crowding that actually exists.
    repair_feedback :
        If the previous attempt failed sandbox/validation, its error message, so
        the model repairs rather than repeats it.
    audit_feedback :
        Set only when a PREVIOUS COMPLETED SEARCH under this same brief was
        rejected by the post-search audit
        (:mod:`pref_dispatch.llm.skill_audit`) because the fitness it authored
        was not asking for the intended behaviour. Carries the rejected formula,
        the audit's arithmetic complaint and the real numbers it produced, so
        generation 0 re-authors the yardstick instead of rediscovering it.
    """
    parts: List[str] = []

    parts.append("# TASK\nDesign ONE lower-layer dispatch skill for the fleet.")
    if objective_hint:
        parts.append(
            f"Target objective for this skill: {objective_hint}\n"
            "Specialise hard -- a single-objective specialist is exactly what we "
            "want here (the upper layer combines specialists later)."
        )
    else:
        parts.append(
            "Choose a single objective that is USEFUL and NOT already covered by "
            "the existing skills below. Specialise hard; do not build a generalist."
        )

    parts.append("# ENVIRONMENT PROFILE\n" + env_profile)

    parts.append("# FUNCTION CONTRACT\n" + SIGNATURE_SPEC)
    parts.append("# AVAILABLE METRICS (for your fitness)\n" + METRIC_MENU)
    parts.append("# DECISION-RULE MECHANISMS\n" + MECHANISM_MENU)
    parts.append("# SANDBOX RULES\n" + SANDBOX_RULES)
    parts.append("# SEED EXAMPLES (valid outputs and your starting point)\n"
                 + few_shot_seeds())

    if existing_skills:
        cards = "\n".join(_skill_card(m) for m in existing_skills)
        parts.append(
            "# SKILLS ALREADY IN THE BASIS (cover a DIFFERENT niche than these)\n"
            + cards
        )

    if repository_note:
        parts.append(
            "# REPOSITORY STATE AND THE DIVERSITY BAR YOU MUST CLEAR\n"
            + repository_note
            + "\nBe ambitious about diversity: the repository is worth more when its "
            "skills disagree. Good moves -- optimise a metric every listed skill "
            "treats as a cost; condition on a driver/order state they all ignore "
            "(near-full vehicles, long-idle drivers, far-flung or low-fare orders, "
            "batching several compatible requests); accept a deliberate loss on the "
            "metric they compete over. Bad moves -- a reweighted blend of the same "
            "terms, or the same objective renamed."
        )

    if similarity_note:
        parts.append(
            "# BEHAVIOURAL DIVERSITY (measured -- push into an uncovered niche)\n"
            + similarity_note
        )

    parts.append(
        "# SELF-CHECK BEFORE YOU SUBMIT (do this EVERY time you write a NEW skill)\n"
        "After writing `code`, mentally re-read your objective and confirm BOTH:\n"
        "  (a) AXIS -- your skill's objective specialises on ONE axis from the list "
        "below (a genuinely different objective is a different CUSTOMER, not a "
        "reworded one).\n"
        "  (b) COVERAGE -- you actually check the existing skills: either your axis "
        "is NOT covered by any listed skill, or you state in `objective_self_check` "
        "why a second specialist on an already-covered axis is still a distinct "
        "behaviour (a different decision rule genuinely aimed at that same axis -- "
        "never just different weights). A repository whose skills all optimise the "
        "same two axes leaves the other axes UNANSWERABLE, and the upper combiner "
        "cannot then respond to a reward that prices one of them: it falls back to "
        "its own blend and WEAKENS against a reward that this skill would have "
        "answered. If you cannot honestly name a new axis, you did not finish the "
        "task -- cover the least-covered axis instead.\n"
        + OBJECTIVE_AXIS_LIST
    )

    if audit_feedback:
        parts.append(
            "# A PREVIOUS SEARCH UNDER THIS SAME BRIEF WAS REJECTED -- READ THIS "
            "BEFORE YOU WRITE THE FITNESS\n" + audit_feedback
        )

    if repair_feedback:
        parts.append(
            "# YOUR PREVIOUS ATTEMPT FAILED -- FIX THIS AND RESUBMIT\n"
            f"{repair_feedback}\n"
            "Return a corrected JSON object addressing exactly this error."
        )

    parts.append("# INTERPRETABILITY\n" + INTERPRETABILITY_RULE)
    parts.append("# OUTPUT FORMAT\n" + OUTPUT_CONTRACT)

    return {"system": SYSTEM_PROMPT, "user": "\n\n".join(parts)}


def build_skill_improve_prompt(
    env_profile: str,
    *,
    objective: str,
    fitness_code: str,
    current_code: Optional[str] = None,
    current_fitness: Optional[float] = None,
    fitness_note: Optional[str] = None,
    parents: Optional[Sequence[Dict]] = None,
    existing_skills: Optional[Sequence[Dict]] = None,
    repair_feedback: Optional[str] = None,
) -> Dict[str, str]:
    """Build the prompt for one *variant* of a skill whose objective is FIXED.

    The objective and the self-authored ``fitness`` are chosen once, in generation
    0; every later call may only rewrite the ``score``/``noop_score`` body to score
    higher under that fixed fitness -- the "agent first designs a reward, then
    trains the optimal skill under it" loop the user asked for. Three shapes, all
    under that same fixed yardstick:

    * **Mutation** (``current_code`` given): improve ONE program. ``fitness_note``
      carries the group-relative read-out (how it stood against the round's other
      variants, and at which fleet scale it was weakest) so the rewrite is aimed at
      a measured weakness rather than at a bare number.
    * **Crossover** (``parents`` given -- a list of ``{name, mechanism, code,
      fitness_note}`` cards): combine TWO programs into one. This is how a variant
      that is the best thing anyone has at fleet 300 passes that mechanism to a
      variant that is strong everywhere else, instead of dying with it.
    * **Parentless** (neither given): write a fresh implementation of the SAME
      objective from scratch. Every mutation/crossover child descends from the
      gen-0 program, so without this slot the whole population converges on one
      template wearing different thresholds.

    ``current_fitness`` is the incumbent's measured fitness, shown for context; it
    is a group-relative advantage, not an absolute score (see ``fitness_note``).
    """
    parts: List[str] = []
    cards = list(parents or [])

    if cards:
        parts.append(
            "# TASK\nCOMBINE the two dispatch skills below into ONE new skill. They "
            "share the same fixed objective and are graded by the same fixed "
            "fitness; each is strong in a different place. Take the mechanism that "
            "makes each one strong and produce a single `score` that keeps BOTH "
            "strengths -- not an average of their coefficients, and not a copy of "
            "whichever looks better."
        )
    elif current_code is not None:
        parts.append(
            "# TASK\nImprove ONE existing dispatch skill. The objective and its "
            "fitness are FIXED; rewrite only the scoring logic to score HIGHER under "
            "that fixed fitness."
        )
    else:
        parts.append(
            "# TASK\nWrite a NEW implementation of the fixed objective below, from "
            "scratch. Do NOT imitate any existing implementation of it: choose a "
            "DIFFERENT decision-rule mechanism (see the mechanism list) and let it "
            "reach the same objective its own way. A genuinely different rule that "
            "scores slightly worse is more useful here than a near-copy that scores "
            "slightly better -- it is the only way a new mechanism can enter."
        )

    parts.append(f"# FIXED OBJECTIVE\n{objective}")
    parts.append(
        "# FIXED FITNESS (do not change; your new code will be scored by this)\n"
        f"```python\n{fitness_code}\n```"
    )

    if cards:
        for i, p in enumerate(cards):
            head = f"# PARENT {i + 1}: {p.get('name', '?')}"
            mech = str(p.get("mechanism", "")).strip()
            if mech:
                head += f"\nmechanism: {mech}"
            note = str(p.get("fitness_note", "")).strip()
            if note:
                head += f"\n{note}"
            parts.append(f"{head}\n```python\n{p.get('code', '')}\n```")
    elif current_code is not None:
        head = "# CURRENT SKILL"
        if current_fitness is not None:
            head += f" (measured fitness = {current_fitness:.4g})"
        parts.append(f"{head}\n```python\n{current_code}\n```")
        if fitness_note:
            parts.append("# HOW IT SCORED (aim your rewrite at this)\n" + fitness_note)

    parts.append("# ENVIRONMENT PROFILE\n" + env_profile)
    parts.append("# FUNCTION CONTRACT\n" + SIGNATURE_SPEC)
    parts.append("# DECISION-RULE MECHANISMS\n" + MECHANISM_MENU)
    parts.append("# SANDBOX RULES\n" + SANDBOX_RULES)

    if existing_skills:
        other = "\n".join(_skill_card(m) for m in existing_skills)
        parts.append("# OTHER SKILLS IN THE BASIS (stay distinct)\n" + other)
    if repair_feedback:
        parts.append(
            "# YOUR PREVIOUS ATTEMPT FAILED -- FIX THIS AND RESUBMIT\n"
            f"{repair_feedback}"
        )

    parts.append("# INTERPRETABILITY\n" + INTERPRETABILITY_RULE)
    parts.append(
        "# OUTPUT FORMAT\n" + OUTPUT_CONTRACT
        + "\nEcho `objective` and `fitness_code` unchanged (both are fixed); we keep "
        "the originals regardless. `mechanism` and `differs_from` must describe "
        "YOUR new code. `objective_self_check` is NOT needed here -- the objective "
        "is already fixed and was checked when it was authored."
    )
    return {"system": SYSTEM_PROMPT, "user": "\n\n".join(parts)}


# Fields the extractor must find non-empty (interpretability enforcement).
REQUIRED_EXPLANATION_FIELDS = ("objective", "description", "fitness_rationale")
# All required fields (explanation + code) for a well-formed skill response.
REQUIRED_FIELDS = (
    "skill_name",
    "objective",
    "description",
    "fitness_code",
    "fitness_rationale",
    "code",
)
# Required ONLY on the group-evolution paths (2026-08-10), which select for
# behavioural diversity and therefore need each program to state the decision rule
# it uses and how that rule differs from what already exists. Kept out of
# REQUIRED_FIELDS so the legacy hill-climb and the Phase-3 warm start -- neither of
# which selects on mechanism -- do not start failing on frozen artifacts that
# predate the field.
REQUIRED_MECHANISM_FIELDS = ("mechanism", "differs_from")

# Required ONLY on the generation-0 authoring path (2026-08-21 self-check). When
# the model writes a NEW objective it must state which objective axis it covers and
# confirm that axis is not already in the repository -- the phase-1 coverage check,
# the analogue of Phase-2's probe self-check. Later variants keep the objective
# FIXED, so this field is not demanded of them; it is enforced via the
# ``require_self_check`` flag in evolve.py `_check_fields`.
REQUIRED_SELF_CHECK_FIELDS = ("objective_self_check",)

# Object-axis labels the self-check must reason over. These are the dimensions the
# skill repository must jointly cover so the Phase-2 combiner always has a
# specialist matching whatever `w` it is handed; a repository that only contains
# "maximise revenue"-style rewordings leaves the service/detour/fairness axes
# unanswerable and the combiner falls back to its own blend.
OBJECTIVE_AXIS_LIST = """\
  revenue / fare       -- favour high-fare (long, group) trips even at a cost
  service / wait       -- minimise passenger waiting and in-car time
  throughput           -- maximise distinct riders served / orders assigned
  detour on new order  -- prefer assignments that add little pooling detour
  remaining capacity   -- use near-full vehicles' last seats for marginal demand
  empty / idle cost    -- avoid empty moves and idle deadheading for the car
  fairness             -- equalise driver take-home income (lift the lowest earner)
  option value         -- wait/patience: hold out for a better match over grabbing
                          the first feasible order
"""

# Software axis detector (2026-08-21). The gen-0 self-check cannot trust the
# model's own prose, so we are teaching the checker to classify WHICH axis an
# objective text specialises on by keyword, then reject a candidate whose axis is
# already saturated in the repository. ``OBJECTIVE_AXES`` gives the canonical
# axis -> matching keyword tuple; ``detect_axis`` maps a short text to the labels
# it mentions (multiple hits are fine; the caller decides how to treat them).
OBJECTIVE_AXES: Dict[str, Tuple[str, ...]] = {
    "revenue": ("revenue", "fare", "high-fare", "trip-minutes", "long fare",
                "valuable trip", "profit"),
    "service": ("service", "wait", "waiting", "in-car time", "service_time",
                "shorter service", "passenger waiting"),
    "throughput": ("throughput", "serve", "coverage", "distinct rider",
                   "orders served", "assigned orders", "unmatched", "accept"),
    "detour": ("detour", "pooling", "extra delay", "re-route", "detour_total"),
    "capacity": ("capacity", "last seat", "remaining seat", "near-full",
                 "marginal load", "topoff", "top-off"),
    "empty_idle": ("empty", "idle", "deadhead", "empty move", "idle wait",
                   "idle_min"),
    "fairness": ("fairness", "income", "gini", "equity", "take-home",
                 "lowest earner", "driver-income"),
    "option": ("option", "patience", "wait for", "hold out", "future match",
               "opportunity cost"),
}


def detect_axis(text: str) -> Tuple[str, ...]:
    """Return the canonical objective axes a candidate's text specialises on.

    Keyword match only -- a cheap, honest classifier. An empty result means the
    text names no known axis (a reworded generalist), which the caller treats as
    a FAILED self-check rather than as a niche."
    """
    low = str(text).lower()
    return tuple(ax for ax, kws in OBJECTIVE_AXES.items()
                 if any(k in low for k in kws))

