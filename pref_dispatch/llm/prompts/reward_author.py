"""Prompt builder for the REWARD-AUTHORING stage (§Phase-2).

Before the upper combiner is composed, a natural-language / weights preference is
TRANSLATED by the LLM into a concrete platform reward function. The model is given
the per-step ``event`` dict it may read (the env's ``RewardFunction`` call
surface), must FIRST explain, in ``reward_understanding``, what the preference
wants and which event quantities it therefore rewards/penalises (a chain-of-thought
interpretability gate), and THEN write ``reward(event) -> float``.

The authored reward is injected into the env as its ``reward_function`` for the
combiner rollouts, so its cumulative fleet mean (``income_mean``) becomes the
combiner's fitness. One preference -> one reward -> one composed strategy: there is
no runtime preference dial here, only a single concrete objective.

v10: the authored reward must be ADDITIVE over the step's events -- see
:data:`pref_dispatch.llm.prompts.common.LINEAR_OBJECTIVE_RULE`, the same block the
brief proposer is given. If the preference text asks for a threshold or a ratio,
the reward encodes the closest additive price list and SAYS SO in
``reward_understanding`` rather than quietly writing the nonlinear version.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pref_dispatch.llm.prompts.common import LINEAR_OBJECTIVE_RULE
from pref_dispatch.llm.reward_spec import event_spec

# Sandbox rules specialised for a reward body over the event dict (no skill
# primitives, no dist/phi -- only the event dict and math/np).
REWARD_SANDBOX_RULES = """\
Code rules (enforced by an AST sandbox -- violating them rejects your reward):
  - `reward(event)` must be a pure function of the single `event` dict. No import
    statements of any kind.
  - Allowed globals: math, np (numpy). Allowed builtins: abs, min, max, sum, len,
    float, int, round, sorted, range, enumerate, zip, map, filter, pow, all, any,
    bool, list, dict, tuple, set.
  - No eval/exec/open/getattr and no attribute access beginning with underscore
    (e.g. no `.__class__`).
  - Index the `assigned_*` sub-dicts only by ids in event['assigned_orders'], and
    use `.get(oid, default)` so an empty step never raises.
  - reward() must ALWAYS return a finite float, including on an idle step where
    event['assigned_orders'] is empty (return 0.0 or a small penalty, never None).
  - The BODY SHAPE is fixed by the additivity rule above: accumulate a per-order
    price in a loop over event['assigned_orders'], add the per-step charges, and
    return the total. `len(...)` is fine as a COUNT to multiply a fixed price by;
    it must not index a table of escalating rates or drive an if/else on volume.
    `min`/`max` may clamp a single order's own quantity (e.g. a detour reading),
    never an accumulated total.
"""

OUTPUT_CONTRACT = """\
Respond with ONE JSON object and nothing else (no prose before/after, no markdown
outside the JSON). Schema:

{
  "reward_understanding": "<2-4 sentences, FIRST: in your own words, what the
                 platform PREFERENCE below wants, and which quantities in the
                 per-step event dict must therefore be rewarded or penalised (and
                 roughly how strongly). This is a chain-of-thought gate: reason
                 about the objective before you write the reward. If the preference
                 asks for something the additivity rule forbids (a threshold, a
                 ratio, an escalating bonus), say here which additive price list you
                 are using instead and what that changes.>",
  "reward_name": "<short snake_case id, e.g. completion_first_reward>",
  "objective":  "<ONE sentence: what a driver maximising this reward will do>",
  "description": "<2-4 sentences: HOW each term of your reward encodes the
                 preference -- what it pays for, what it charges, and the relative
                 magnitudes. Explain behaviour, not code.>",
  "code": "def reward(event):\\n    ..."
}

The reward is a per-driver, per-step scalar. It is summed over the whole episode
and averaged across the fleet; that fleet-mean cumulative value is THE objective a
dispatcher is then composed to maximise. Make the term magnitudes reflect the
stated preference (e.g. a throughput-first preference makes the assignment term
dominate).
"""

# How to ESTIMATE each common metric from the per-step event dict. This is what
# makes an authored reward express a metric the interpreter would otherwise leave
# unmeasured. Pick the terms the preference asks for; DIFFERENT preferences need
# different subsets -- do not force every metric into one reward.
_METRIC_ESTIMATION = """\
# HOW TO ESTIMATE EACH METRIC (pick the terms your preference actually asks for)
The env hands each driver-step an `event` dict. For a newly assigned order `oid`,
the passenger's journey splits into THREE wait/ride components plus pooled detour.
Each is a separate priced term; price them independently, and optionally weight any
by `assigned_party_sizes[oid]` (a group waiting is costlier than a solo rider):

  dispatch_wait   assigned_dispatch_wait[oid]        # minutes on the platform before
                                                    # the driver was sent (waiting).
  pickup_time     assigned_pickup_times[oid]         # minutes the driver travels to
                                                    # reach the pickup point.
  solo_time       assigned_solo_times[oid]           # direct pickup->dropoff ride time.
  ride_detour     assigned_detour_times[oid]         # extra END-TO-END delivery time from
                                                    # pooling: pooled drop-off minus (direct
                                                    # driver->pickup + solo ride). Includes
                                                    # driver detour AND in-vehicle detour; >= 0.
  service_time    assigned_service_times[oid]        # dispatch_wait + pickup + ride
                                                    # (end-to-end); always >= solo.
  throughput      len(event['assigned_orders'])      # orders accepted this step.
  revenue         solo_times[oid] * assigned_party_sizes[oid]   # fare proxy.
  completion      completed_orders                    # a drop-off actually landed.
  empty/idle      is_empty_move / is_idle_wait        # wasted movement / sitting.

DELTA RULE: every term is a CHANGE, not a static value. Accepting a new order may
alter ALREADY-ONBOARD orders too (their route changed, so their pickup/delivery
times move). For every term, consider BOTH the new order's own value AND the signed
change to each onboard order -- a later delivery is a positive cost, an earlier one
a negative one. A reward may price the new-order and onboard-order parts with
DIFFERENT coefficients, so keep them separate.

The additivity rule fixes the SHAPE (a per-order price list plus per-step charges).
Within it, pick the subset of terms the preference points at and weight them. A
reward about passenger waiting prices dispatch_wait and/or pickup_time (optionally
weighted by party size); one about pooling prices assigned_detour_times or
party_size; one about waste prices empty/idle. A batch of objectives should contain
single-metric ones as well as blends, so each axis is learned separately.
"""

SYSTEM_PROMPT = """\
You are an expert in ride-POOLING fleet economics and reinforcement-learning
reward design. You are given a platform PREFERENCE (natural language or weights)
and the exact per-step event signals available. You must first explain, in
natural language, what that preference wants, and then write ONE small, robust,
interpretable Python reward function `reward(event) -> float` that encodes it. You
always answer with exactly one JSON object matching the requested schema.
"""


def build_reward_prompt(
    preference_spec: str,
    *,
    current_code: Optional[str] = None,
    current_fitness: Optional[float] = None,
    repair_feedback: Optional[str] = None,
    stage: str = "train",
    probe_spec: str = "",
) -> Dict[str, str]:
    """Build the ``{"system","user"}`` prompt for the reward-authoring stage.

    ``preference_spec`` is the rendered platform preference (from
    :func:`pref_dispatch.llm.reward_spec.reward_spec`): a natural-language brief, a
    weight-vector statement, or a described concrete reward. ``current_code`` /
    ``current_fitness`` are ``None`` on the first (propose) generation and set on
    improvement generations (rewrite the reward to raise its realised fleet mean).

    ``stage`` picks the authoring contract (see :func:`author_reward`):
    ``"train"`` translates the preference faithfully (the objective is ground
    truth); ``"infer"`` writes it so the frozen probe geometry can detect it.
    ``probe_spec`` is the stage-dependent probe/probe-probe text rendered below.
    """
    parts: List[str] = []

    if current_code is None:
        parts.append(
            "# TASK\nTranslate the platform PREFERENCE below into ONE concrete "
            "per-driver, per-step reward function `reward(event) -> float`. First "
            "explain what the preference wants, then write the reward. This reward "
            "becomes the FIXED objective a dispatcher is later composed to maximise."
        )
    else:
        parts.append(
            "# TASK\nImprove your reward so that, when a dispatcher maximises it, "
            "the fleet behaviour better matches the platform PREFERENCE below. Keep "
            "the same contract and re-explain the (improved) encoding."
        )
        if current_fitness is not None:
            parts.append(
                "# CURRENT REWARD (measured fleet-mean cumulative value = "
                f"{current_fitness:.4g})\n```python\n{current_code}\n```"
            )

    parts.append("# PLATFORM PREFERENCE (translate this into a reward)\n" + preference_spec)
    parts.append("# EVENT DICT (the only signals your reward may read)\n" + event_spec())
    parts.append("# HOW TO ESTIMATE EACH METRIC\n" + _METRIC_ESTIMATION)
    parts.append("# WHAT A REWARD MAY BE\n" + LINEAR_OBJECTIVE_RULE)
    parts.append("# SANDBOX RULES\n" + REWARD_SANDBOX_RULES)

    if probe_spec:
        parts.append("# WHY/HOW YOU ARE WRITING THIS REWARD\n" + probe_spec)

    if repair_feedback:
        parts.append(
            "# YOUR PREVIOUS ATTEMPT FAILED -- FIX THIS AND RESUBMIT\n"
            f"{repair_feedback}\n"
            "Return a corrected JSON object addressing exactly this error."
        )

    parts.append("# OUTPUT FORMAT\n" + OUTPUT_CONTRACT)

    return {"system": SYSTEM_PROMPT, "user": "\n\n".join(parts)}


# Interpretability fields the extractor must find non-empty. ``reward_understanding``
# is the CoT gate: the model must show it read the preference before authoring.
REQUIRED_EXPLANATION_FIELDS = ("reward_understanding", "objective", "description")
# All required fields for a valid authored reward.
REQUIRED_FIELDS = ("reward_understanding", "reward_name", "objective", "description", "code")
