"""Prompt builder for Phase-2 upper-combiner evolution (§5.2; final signature).

The lower skills are FROZEN. Here the model writes a single objective-reading
function (final-version two-layer signature)::

    skill_scores(driver_obs, phi_ep, phi_step, w) -> {skill_name: score}

that, for one driver, scores each frozen skill; the platform blends that driver's
top ``DEFAULT_BLEND_K`` skills (softmax over the scores, each skill standardized
across the driver's candidates first) into its scorer this step. The headline claim is *zero-retrain objective
adaptation*: the SAME frozen function must serve ANY episode objective it never
saw, because it reads the objective ``w`` (an LLM-authored callable reward
function, or ``None``) as an INPUT rather than being tuned for one operating
point. Skills stay objective-blind specialists; only the combiner (and the
repositioner) see ``w``, so the combiner is the layer that specialises the BLEND
to the objective without the skill library collapsing into one maximiser.

Two-layer context (final-version redesign):
  * ``phi_ep`` -- episode-STATIC (fleet/capacity/speed, region layout,
    ``phi_ep.dist`` travel-time closure, static ``phi_ep.scale``).
  * ``phi_step`` -- LIVE per-step (``demand_pressure``, ``mean_solo_time``, ...).

Unlike Phase 1, the model does NOT author a fitness -- the Phase-2 fitness is
fixed by the researcher (§5.4), so this prompt has no ``fitness_code`` field.
Interpretability still holds: the model must explain, in natural language, how it
maps driver state x objective to a skill choice.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from pref_dispatch.llm.prompts.common import PHI_FIELDS_SPEC, SANDBOX_RULES

# The output contract. Explanation fields are FIRST and called out as required.
OUTPUT_CONTRACT = """\
Respond with ONE JSON object and nothing else (no prose before/after, no markdown
outside the JSON). Schema:

{
  "combiner_name": "<short snake_case id, e.g. objective_aware_dispatcher>",
  "strategy":      "<ONE sentence: the overall policy for turning the objective +
                    driver state into a skill choice>",
  "description":   "<3-5 sentences: which driver STATES you distinguish (idle /
                    loaded-with-slack / deadline-pressed / ...), which skill each
                    tends to get, and HOW the episode objective shifts those choices.
                    Explain behaviour, not code.>",
  "code": "def skill_scores(driver_obs, phi_ep, phi_step, w):\\n    ...\\n    return {<skill>: <score>, ...}"
}
"""

# v2+: when the episode objective is given as an explicit REWARD FUNCTION (not just
# an abstract handle), the model must FIRST explain that reward in its own words --
# a chain-of-thought gate (mirrors the v1 pure-Phase-1 arm) -- before composing the
# skills FOR it. Adds a mandatory ``reward_understanding`` field, kept first.
REWARD_OUTPUT_CONTRACT = """\
Respond with ONE JSON object and nothing else (no prose before/after, no markdown
outside the JSON). Schema:

{
  "reward_understanding": "<2-4 sentences, FIRST: in your OWN words, what the
                    REWARD FUNCTION above rewards and penalises, and what a
                    reward-maximising dispatcher must therefore do. This is a
                    chain-of-thought gate: reason about the objective BEFORE you
                    compose the skills.>",
  "combiner_name": "<short snake_case id, e.g. reward_aware_dispatcher>",
  "strategy":      "<ONE sentence: how you turn that reward + driver state into a
                    skill choice>",
  "description":   "<3-5 sentences: which driver STATES you distinguish (idle /
                    loaded-with-slack / deadline-pressed / ...), which skill each
                    tends to get, and HOW the reward's terms (throughput / revenue /
                    service / detour) drive those choices. Explain behaviour, not code.>",
  "probe_self_check": "<2-4 sentences: CONFIRM your probes are diverse enough. List
                    which of the KNOWN terms your probes can detect (dispatch_wait,
                    pickup_time, solo_time/service_time, detour on a new order,
                    detour on onboard orders, completion, seating/party, volume,
                    empty/idle) -- and which EXTRA terms you added that the reward
                    MIGHT price. State explicitly: probes whose term turned out absent
                    from THIS reward are still kept, because a different reward the
                    same combiner must read may price them; probe COUNT is cheap and
                    the combiner need not use all of them. If a listed term is not
                    probed, say why and justify it -- do not silently drop it.>",
  "code": "def skill_scores(driver_obs, phi_ep, phi_step, w):\\n    ...\\n    return {<skill>: <score>, ...}"
}
"""

SYSTEM_PROMPT = """\
You are an expert in ride-POOLING fleet dispatch and objective-conditioned policy
design. A set of lower-layer scoring skills is ALREADY FROZEN; you cannot change or
add skills -- you only decide, per driver and per episode objective, WHICH frozen
skill that driver should use. You write one small, robust, interpretable Python
function AND a clear natural-language explanation of the policy. You always answer
with exactly one JSON object matching the schema.
"""

# How the objective `w` reaches the combiner and how to read it.
OBJECTIVE_SPEC = """\
`w` is the EPISODE OBJECTIVE handed to your combiner every step. It is EITHER:
  - a callable reward function `w(event) -> float` (LLM-authored, frozen for the
    whole episode) that grades one driver-step `event`, OR
  - `None` (objective-blind), in which case fall back to a sensible balanced blend
    driven by driver state and `phi_step.demand_pressure`.
You MAY call `w` on a small probe `event` dict to self-derive which frozen skill
best serves the objective for a given driver -- the `reward_aware_dispatcher_v2`
pattern -- but you MUST guard the call (wrap in try/except or check `w is not None`)
so a `None` or ill-behaved `w` never crashes the step. Do NOT assume any particular
reward shape: read what `w` returns, do not hard-code coefficients.

An `event` dict you may probe `w` with has these keys (values may be empty):
  assigned_orders(list[int]), assigned_party_sizes(dict), assigned_solo_times(dict),
  assigned_service_times(dict), completed_orders(list), picked_up_orders(list),
  distance_moved(float), time_moved(float), is_empty_move(bool), is_idle_wait(bool),
  extra_detour_time(float). Build probe events from THIS driver's candidate orders
  (e.g. a would-be long-fare assignment vs a short one) to see what the objective
  prefers, then route the driver to the skill that pursues it.
"""

# v4: the objective's TERM STRUCTURE varies across episodes (linear mixes,
# completion-gated, pooling/seating, empty/idle-averse), so the combiner must read
# WHAT KIND of event `w` rewards, not just one scalar. Injected right after
# OBJECTIVE_SPEC so the probe vocabulary is directly above it.
# v10: the progressive/nonlinear family is gone from the objective distribution --
# every `w` is now LINEAR in the per-step event terms -- so it is no longer
# advertised here.
OBJECTIVE_DIVERSITY_SPEC = """\
THE OBJECTIVE'S TERM STRUCTURE VARIES, NOT JUST ITS COEFFICIENTS. Across episodes
`w` is drawn from a distribution of genuinely DIFFERENT reward families. Every one
of them is LINEAR in the per-step event terms -- a fixed price per assignment, per
drop-off, per seat, per minute of service, per unit of detour -- but WHICH terms
carry the price changes:
  - linear coefficient mixes over assignment / revenue / service-time / detour /
    empty-move / idle penalties (the classic shaped reward);
  - COMPLETION-gated rewards that pay on DROP-OFF (`completed_orders`) -- an
    assignment only earns when it is actually finished;
  - POOLING / seating rewards that pay PER PASSENGER (`assigned_party_sizes`) --
    filling an empty seat is worth real money; and
  - empty/idle-AVERSE rewards that punish a car that is not producing.
Because every `w` is linear, its price per unit of a term is CONSTANT: probing the
same term at two different magnitudes tells you the coefficient, and a probe with
twice the volume returns twice the value. You can rely on that.
The SAME probe event therefore returns very different values under different
objectives, and a combiner that only reads "long fare vs short fare" will be
outcompeted. Probe the event SHAPE, not one scalar: vary whether the probe carries
completed orders (completion-gated?), the party size (seating?), the service time
and detour (how are they priced?), and the empty/idle flags (aversion?), and read
WHAT KIND of event the objective rewards -- then blend the skills toward whatever
pursues it. Guard every probe; fall back to driver state + demand_pressure when
`w` is None or unusable.

THE STRATEGY MUST CHANGE WITH THE OBJECTIVE FUNCTION -- THIS IS THE HEADLINE
REQUIREMENT. You are NOT writing one policy that is "good on average": you are
writing a policy that SPECIALISES PER OBJECTIVE. The objective ``w`` is the input
that decides the strategy, every step. A fleet whose skill mix is IDENTICAL
under a completion reward and a pooling reward is a FAILURE no matter how high its
average reward -- it is not doing the job, and the selection below ranks it as
blind. As a guide, the objective's TERM STRUCTURE should push the blend like this:
  - COMPLETION-gated ``w`` (pays on ``completed_orders`` / drop-off) -> lean the
    fleet toward a RELIABLE fast-serve, low-detour specialist (an assignment only
    earns when it is actually finished);
  - POOLING / seating ``w`` (pays per passenger in ``assigned_party_sizes``) ->
    lean toward the POOLING specialist and fill spare seats;
  - assignment-VOLUME ``w`` (a flat, generous price per order accepted) -> lean
    toward the BROAD-COVERAGE specialist (never leave a reachable rider behind);
  - length-driven ``w`` (long fares >> short) -> lean toward the LONG-FARE /
    revenue specialist, and let loaded-with-slack cars chase it;
  - empty/idle-AVERSE ``w`` (punishes a car that is not producing) -> the fleet
    must never sit idle: lean toward coverage/quick-service even at high demand.
The per-driver state still decides WITHIN an objective (a deadline-pressed car
protects its onboard order regardless); the objective decides ACROSS objectives.
Write the probes so that under structurally different ``w`` values the skill
MIX GENUINELY shifts, and say in ``description`` which objective shapes drive
which skill choices. When in doubt about the shape, probe ``w`` directly on the
event kinds above rather than guessing.

YOU ARE SCORED ON WHAT YOUR CHOOSING IS WORTH, PER OBJECTIVE -- NOT ON RAW REWARD.
Your candidate is rolled on every (scene, objective) pair of THIS round, and scored
on each pair as

    (YOUR episode reward - the EQUAL BLEND's episode reward on the SAME scene and
     seed) / how much this round's programs disagree about that same difference.

The EQUAL BLEND is the do-nothing upper layer: every frozen skill given the same
weight, for every driver, at every step, under every objective. It still
dispatches -- it just never CHOOSES. So the numerator is exactly the money your
choosing earned, and the sign is absolute:
  - 0.00 means your per-driver skill choice was worth exactly as much as not
    choosing at all;
  - NEGATIVE means choosing ACTIVELY LOST money against a flat blend -- you steered
    drivers onto skills that paid less than an unweighted mix;
  - +1 means your gain over not choosing is one full spread above how much the
    other programs' gains vary.
Reward scales do NOT matter: an objective whose numbers are 1000x smaller counts
exactly the same as one 1000x larger. Beating the other candidates is NOT the
target -- BEATING "NO CHOICE WAS MADE" IS. If nothing you can think of beats the
equal blend on a family, the honest answer is a blend close to equal there, not a
confident wrong pick. What DOES matter beyond the sign is the MARGIN: beating the
baseline by a lot on one objective is worth more than nosing past it on three. The
fitness note tells you the per-family breakdown -- the fastest way to score higher
is to turn your NEGATIVE and near-zero families positive, not to squeeze the ones
you already win.

CRASHING SCORES EXACTLY 0, AND IT IS NOT A HIDING PLACE. If your `skill_scores`
raises, returns a non-dict, or names no known skill, the platform runs the equal
blend for that driver -- the baseline itself -- so a program that breaks everywhere
scores 0.00, the same as one that ran perfectly and chose nothing useful. There is
no penalty coefficient and no credit borrowed from some default skill. You are also
given exactly ONE repair attempt with the real error text; a program that still
breaks after it is ELIMINATED from the round regardless of its score.

THE SCENES CHANGE EVERY ROUND. Each round draws a fresh batch of real one-hour
demand windows and a fresh batch of objectives, and every surviving parent is
re-rolled on them alongside you -- so a policy that only wins because it was lucky
on last round's scenes does not survive. Nothing you write may depend on a
particular fleet size, hour, or demand level; it must win on scenes it has not
seen, under objectives it has not seen.

SELECTION ADDS A BONUS FOR YOUR WEAKEST FAMILY. The selection key is NOT the plain
mean: it is ``mean advantage + 0.15 x YOUR WEAKEST-family advantage``. Every family
your candidate lags therefore costs DOUBLE (once in the mean, once through the
weakest-family term) -- raising the family you lose most on is the fastest way to
be selected, so a candidate that reads EVERY objective family, even imperfectly,
beats one that ignores one family entirely. On top of that, each objective family
keeps a RESERVED SURVIVOR SLOT: the best program on a family survives the round
even if its overall mean is mediocre, so genuinely specialising on a hard family
is a way to stay alive and be built on.

A WRONG FLIP IS WORSE THAN NO FLIP. This is now literal, not rhetorical: if you
steer the fleet onto a skill that pays LESS at that operating point than an
unweighted blend would have, your score on that pair goes NEGATIVE -- below the
0.00 a program that never chose anything gets. When you cannot tell whether a
switch pays, prefer the conservative switch (or no switch). This is not a penalty
term -- it is simply what "your choosing lost money" looks like when the baseline
is not choosing.

RESPONDING DIFFERENTLY TO DIFFERENT OBJECTIVES IS MEASURED, NOT ASSUMED. Your
fitness note reports, per objective family, what fraction of the round produced
the EXACT SAME episode reward as you (the "DEAD FAMILIES" line), and how many of
the frozen single skills beat you there. Read it literally:
  - A family where you are identical to most of the round means NO program --
    including yours -- changed anything when the objective took that shape. An
    advantage near 0.00 there is not "average", it means your choosing was worth
    nothing over an equal blend, and single skills are already clearing that bar.
  - The previous run failed exactly this way: across five families its fleet had
    only TWO distinct behaviours -- one for seating/pooling rewards and one for
    everything else. Completion-gated, progressive and linear objectives all got
    byte-identical dispatch, i.e. on three of the five families it was running one
    fixed policy that happened not to be the equal blend, and nothing it did
    depended on what it was asked to maximise.
So do not write one policy with a single special case bolted on. Give the
COMPLETION-gated shape, the PROGRESSIVE shape, the empty/idle-AVERSE shape and the
length-driven shape each their OWN observable consequence for which skill a driver
gets -- and make sure a car in the same state, in the same scene, under two
differently-shaped ``w`` values, can end up on DIFFERENT skills. If your probes
cannot tell two of those shapes apart, add a probe that can (vary completed
orders, party size, order COUNT, and the empty/idle flags independently and
compare what ``w`` returns) rather than collapsing them onto one branch.
"""

from pref_dispatch.matching import DEFAULT_BLEND_K

OBJECTIVE_CONTRACT = f"""\
Your function MUST have this exact signature (do not change it):

    def skill_scores(driver_obs, phi_ep, phi_step, w) -> dict:
        # return {{skill_name: score}} scoring the FROZEN skills for THIS driver.

Argument contract (read-only; never mutate them):
  driver_obs["self"]: dict with fields
    location=(lon,lat), current_region(int), status(str), capacity(int),
    committed_passengers(int),
    assigned_order_details=[{{order_id, origin=(lon,lat), destination=(lon,lat),
                             num_passengers, onboard(bool), eta(minutes)}}]
  driver_obs["pending_orders"]: list of the orders visible to this driver, each
    a dict with order_id, origin=(lon,lat), destination=(lon,lat),
    origin_region(int), destination_region(int), num_passengers,
    waiting_time(minutes).
{PHI_FIELDS_SPEC}
  `w` is described in the OBJECTIVE section above.

HOW YOUR SCORES ARE USED. The platform takes your {{skill}}->{{score}} dict, keeps the
{DEFAULT_BLEND_K} highest-scoring skills with a positive score, softmax-normalises those
scores into weights, standardises each skill across this driver's candidate orders
(so skills on different raw scales mix fairly), and dispatches on the weighted
sum. So:
  - your RELATIVE scores matter, not just which one is largest -- a runner-up you
    score close to the leader really does shape the decision, one you score far
    below it barely does;
  - scoring exactly one skill and zeroing the rest is legal and gives the old
    one-skill-per-driver behaviour, but it throws away the blend: prefer naming a
    primary plus the supports that serve the same objective, e.g. a
    completion-shaped `w` might get fast-serve 1.0, coverage 0.6, revenue 0.1;
  - the blend is per DRIVER per STEP, so different cars in the same fleet can carry
    different mixes in the same step.

Rules:
  - Only use the frozen skill names listed below as keys. Any other key is
    invalid and your combiner will be rejected.
  - Score at least one known skill for every driver (the blend must be defined).
  - Distinguish drivers by STATE: read driver_obs["self"] -- an empty
    assigned_order_details means an idle/empty car; onboard orders with a small
    `eta` mean a deadline-pressed car; comfortable eta means loaded-with-slack.
    Route each state to whichever frozen skill best serves the episode objective.
  - Read the objective through `w` (see the OBJECTIVE section) and the live scene
    through `phi_step` (demand_pressure, mean_solo_time, per-region demand/supply);
    never hard-code a fixed operating point. The SAME function must serve
    objectives it has never seen.
  - Handle `w is None` gracefully (balanced fallback). Handle a driver with no
    nearby orders gracefully (still return a defined blend).
"""

# v2: the deployment scene (fleet size / capacity / speed / demand) is randomized
# too, so the combiner must be SCALE-INVARIANT -- read the scene from phi_step /
# phi_ep and the driver's own state, never from a hard-coded fleet size or distance.
SCENE_VARIABILITY_MANDATE = """\
THE DEPLOYMENT SCENE ALSO CHANGES (write a SCALE-INVARIANT combiner). At test
time the SAME frozen function is run zero-shot on scenes that differ in:
  - fleet size (~100 to ~2000 drivers),
  - per-vehicle capacity (~1 to ~10 seats),
  - driver speed, and
  - demand / time-of-day (see the DEMAND section of the profile: prev-1h and
    prev-2h order volume and the passenger-count mix).
So do NOT hard-code any absolute number tied to one scene -- no fixed fleet size,
no absolute distance/time constant, no "if drivers > 500". Express every threshold
RELATIVELY:
  - use `phi_step.mean_solo_time` (fall back to `phi_ep.scale` when it is ~0) as
    the time yardstick;
  - read THIS driver's remaining capacity from driver_obs["self"] (onboard count
    vs seats) rather than assuming a capacity;
  - judge congestion/scarcity from `phi_step.demand_pressure`, not from the raw
    fleet count.
A combiner that reads the scene through these relative signals transfers to a new
fleet/capacity/demand with ZERO retraining -- which is exactly what is tested.

DIFFERENT SCENES MAY DEMAND DIFFERENT STRATEGIES UNDER THE SAME OBJECTIVE. A
300-car fleet in a peak hour is a SCARCITY regime; a 2000-car fleet off-peak is an
OVERSUPPLY regime. The same reward that favours picky long-fare hunting when cars
are plentiful wants broad coverage and quick service when cars are scarce (there
is nobody to be picky with), and within ONE step the same objective may justify a
different skill for a loaded deadline-pressed car than for an idle empty one. Read
the scene (`phi_step.demand_pressure`, this driver's slack and spare seats) AND
the objective `w` TOGETHER, every step; neither one alone decides the blend. The
question per driver is: "given THIS car's state in THIS scene, which skill best
pursues THIS objective?"
"""


def _skill_card(meta: Dict) -> str:
    """A frozen skill's card: name + objective + behaviour, for the combiner to
    reason about which skill fits which driver/objective (§5.2)."""
    name = meta.get("skill_name", meta.get("name", "?"))
    obj = meta.get("objective", "(objective not recorded)")
    desc = meta.get("description", "")
    sig = meta.get("signature_text", "")
    card = f"- {name}: {obj}"
    if desc:
        card += f"\n    behaviour: {desc}"
    if sig:
        card += f"\n    signature: {sig}"
    return card


FEWSHOT_COMBINER = """\
A valid, runnable seed combiner (the handwritten heuristic in the skill_scores
form). Study how it branches on driver STATE (never on a hard-coded operating
point), reads the live scale from phi_step, and consults the objective `w` through
a guarded probe so it adapts to an unseen objective while still working when
`w is None`. Keep this shape:

```python
def skill_scores(driver_obs, phi_ep, phi_step, w):
    self_obs = driver_obs["self"]
    details = self_obs["assigned_order_details"]
    etas = [d["eta"] for d in details if d.get("eta") is not None]
    min_slack = min(etas) if etas else None
    m = float(phi_step.mean_solo_time) or float(phi_ep.scale) or 1.0

    if min_slack is not None and min_slack <= 0.7 * m:
        # deadline-pressed: protect the onboard order regardless of objective.
        return {"enroute": 1.0, "service": 0.3, "revenue": 0.0}
    if min_slack is not None:
        # loaded with slack: mostly service, a little revenue appetite.
        return {"service": 0.7, "enroute": 0.2, "revenue": 0.3}

    # idle/empty car: let the objective decide the revenue<->service lean. Probe
    # `w` (guarded) with a long-fare vs short-fare would-be assignment to see which
    # the objective values more for THIS car; fall back to demand pressure if w is
    # None or unusable. Probe SHAPE, not just length: the same `w` may pay on
    # COMPLETED drop-offs, on FILLED SEATS (party size), or on plain assignment
    # volume -- and may penalise empty moves or long service times. When those
    # differ, probe a completed vs an assignment-only event, a party-2 vs party-1
    # event, a long-service vs short-service event, and lean toward whichever the
    # objective actually rewards.
    loc = self_obs["location"]
    pend = driver_obs.get("pending_orders", []) or []
    lean = 0.5 + 0.2 * float(phi_step.demand_pressure)  # scarce supply -> chase money
    if w is not None and pend:
        def d2(a, b):
            return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
        rides = [d2(o["origin"], o["destination"]) ** 0.5 for o in pend]
        avg = sum(rides) / len(rides) if rides else 1.0
        long_o = max(pend, key=lambda o: d2(o["origin"], o["destination"]))
        short_o = min(pend, key=lambda o: d2(o["origin"], o["destination"]))
        try:
            def ev(o):
                pickup_d = d2(loc, o["origin"]) ** 0.5
                solo_d = d2(o["origin"], o["destination"]) ** 0.5
                # End-to-end service = dispatch wait (a nonzero constant) + pickup
                # travel + in-vehicle ride, so service_time > solo_time always.
                dw = 0.6 * m   # a realistic platform-/dispatch-wait proxy (minutes)
                return {
                    "assigned_orders": [o["order_id"]],
                    "assigned_party_sizes": {o["order_id"]: o["num_passengers"]},
                    "assigned_dispatch_wait": {o["order_id"]: dw},
                    "assigned_pickup_times": {o["order_id"]: pickup_d},
                    "assigned_solo_times": {o["order_id"]: solo_d},
                    "assigned_service_times": {o["order_id"]: dw + pickup_d + solo_d},
                    "assigned_detour_times": {o["order_id"]: 0.0},
                    "completed_orders": [], "picked_up_orders": [],
                    "distance_moved": 0.0, "time_moved": 0.0,
                    "is_empty_move": False, "is_idle_wait": False,
                    "extra_detour_time": 0.0,
                }
            gain_long = float(w(ev(long_o)))
            gain_short = float(w(ev(short_o)))
            span = abs(gain_long) + abs(gain_short) + 1e-9
            lean = 0.5 + 0.5 * (gain_long - gain_short) / span  # 0..1
        except Exception:
            pass
    lean = min(1.0, max(0.0, lean))
    return {
        "revenue": lean,
        "service": 1.0 - lean,
        "enroute": 0.1,
    }
```
"""


def _skill_cards_block(frozen_skills: Sequence[Dict]) -> str:
    names = ", ".join(m.get("skill_name", m.get("name", "?")) for m in frozen_skills)
    cards = "\n".join(_skill_card(m) for m in frozen_skills)
    return (
        f"The FROZEN skills you may select from (use EXACTLY these names as keys): "
        f"{names}\n\n{cards}"
    )


# ======================================================================
# PROBE EVENT EVOLUTION SPEC (optional; activated by --probe-event-evolve)
# ======================================================================
PROBE_EVENT_EVOLVE_SPEC = """\
PROBE EVENT DESIGN IS PART OF YOUR EVOLVING POLICY.
Reward functions are open-ended and may price ANY combination of additive
per-event signals. You must design probe events that can detect the full
shape of whatever reward ``w`` actually prices.

CRITICAL DEFINITION: EVERY reward term is a DELTA, not a static value.
At a decision step the vehicle may already be carrying one or more orders.
Consider accepting a NEW order ``o`` in this step. Each term below is the
CHANGE in the corresponding system-estimated quantity caused by accepting
``o``, relative to the current planned routes (the "before" state). The
passenger's journey splits into SEPARATE wait/ride components (probe each):

  dispatch_wait_term = CHANGE in the minutes a passenger spent on the platform
                       before the driver was sent (now - request_time). Usually
                       only o's own wait; the re-plan does not change it, so
                       this is a static per-o value, but it is a DISTINCT term
                       from pickup time.
  pickup_time_term   = [driver's added travel time to reach o's pickup point] (a)
                       [minus] the change in OTHER orders' pickup/arrival times
                       caused by re-planning (b), passenger-weighted.
  solo_time_term     = o's direct pickup->dropoff ride time (unaffected by other
                       orders); a static per-o value but a DISTINCT fare proxy.
  service_time_term  = [sum of predicted end-to-end service times of ALL orders
                       onboard after accepting o] - [sum before]. This includes
                       o itself AND any already-onboard order whose route
                       changed. NOT just o's own value.
  ride_time_term     = same delta for ride time: after - before, all orders.
  detour_time_term   = [o's own extra END-TO-END delivery time from pooling:
                       its pooled drop-off time MINUS (direct driver->pickup
                       travel + o's solo ride)] PLUS [the change in any
                       already-onboard order's ride time due to o]. The reward
                       may weight the new-order part and the onboard-order part
                       DIFFERENTLY -- you must probe each separately and read
                       the two coefficients independently.
  completion_term    = change in the number (or probability) of orders that
                       will complete after accepting o, vs before.
  seating_term       = change in total passenger-seats occupied / weighted
                       by party size, over ALL affected orders, vs before.
  volume_term        = change in the number of orders carried this step, vs
                       before (usually +1 for o alone).
  empty/idle_term    = change in the vehicle's empty-move / idle-wait time.

WHY DELTAS: each step's indicators are ESTIMATES from the current shortest
route, not ground truth. A static value like "o's solo time" is not what the
reward prices; the reward prices HOW ACCEPTING o CHANGES THE SYSTEM. So the
probe events must encode deltas.

MANDATORY PROBE RULES (you must follow these; skipping them is not allowed):

1) For every term you probe, create at least TWO events that differ ONLY in
   that term's value (one with the delta present, one without), so you can
   compute w(with_term) - w(without_term).

2) You MUST create at least one probe where the vehicle ALREADY has one or
   more orders onboard (``picked_up_orders`` non-empty) AND receives a NEW
   assignment (``assigned_orders`` non-empty) with a NONZERO
   ``extra_detour_time`` -- the bundling case. This lets the combiner detect
   how accepting a new order detours the already-onboard passengers.

3) You MUST create at least one probe where ``picked_up_orders`` is empty
   and the vehicle gets a single new order, with the onboard detour zero --
   the non-bundling baseline.

4) You MUST create at least one probe where
   ``assigned_dispatch_wait[oid]`` is non-zero and one where it is zero, to
   detect the dispatch-wait price; and at least one probe where
   ``assigned_pickup_times[oid]`` differs substantially from ``solo_time``
   (a long pickup vs a short one) to detect the pickup-time price. The two are
   SEPARATE terms (one is platform wait, the other is the driver's travel).

5) You MUST create at least one probe with ``completed_orders`` non-empty and
   one with it empty, to detect the completion delta.

6) You MUST create at least one probe with ``assigned_party_sizes`` > 1 and
   one with party_size = 1, to detect the seating delta.

7) For each probe, make sure the event dict distinguishes NEW orders (in
   ``assigned_orders``) from ALREADY-CARRIED orders (in
   ``picked_up_orders``). The two groups may be priced differently by the
   reward, and only by separating them can the combiner read both weights.

DETECTION METHOD:
After calling w on each probe, compute DIFFERENCES between events that
differ only in one term. For example:
  detour_onboard_signal = w(bundling_with_detour) - w(bundling_without_detour)
  detour_new_signal     = w(new_order_with_detour) - w(new_order_without_detour)
  dispatch_signal       = w(dispatch_wait_nonzero) - w(dispatch_wait_zero)
  pickup_signal         = w(long_pickup) - w(short_pickup)
  completion_signal     = w(completion) - w(no_completion)
  seating_signal        = w(party_2) - w(party_1)
These differences tell the combiner the reward's coefficient per term -- and,
for detour, the SEPARATE coefficient on new-order detour vs onboard detour.

DESIGN PRINCIPLES (no hard-coded event count -- you decide how many are needed):
  - Every reward term must be the DIFFERENTIATING factor in at least one probe.
  - Vary MULTIPLE fields together in at least one probe (e.g. a new order +
    a nonzero detour while carrying a passenger) so interactions are visible.
  - Use realistic magnitudes drawn from the problem domain (solo_time, detour,
    pickup_wait, party_size).
  - BOTH onboard-order and new-order terms must be represented; the reward may
    weight them differently.
  - PROBE COUNT CAN BE GENEROUS. You are probing a UNKNOWN reward, so you do not
    know in advance which terms it prices. Design MORE probes than you expect to
    use: one for each term you can imagine the reward containing. A probe whose
    term turns out to be ABSENT from THIS reward is NOT useless -- that same term
    may be precisely what a DIFFERENT reward function prices, and the same frozen
    combiner must read that other reward too. So an extra probe costs you nothing
    and buys robustness across objectives. Do not prune probes just because the
    current reward does not move on them.

SELF-CHECK BEFORE YOU SUBMIT (do this EVERY time you write a combiner):
After writing your `code`, mentally re-read its probes and confirm BOTH:
  (a) COVERAGE -- each of these KNOWN terms is the differentiating factor in at
      least one probe (or you explicitly justify dropping it in
      `probe_self_check`): dispatch_wait, pickup_time, solo_time/service_time,
      detour on a NEW order, detour on ONBOARD orders, completion, seating/party
      size, volume, empty-move, idle-wait. A reward may price ANY of these, and
      the same frozen combiner must read ANY future reward.
  (b) EXPLORE -- you added probes for terms the reward MIGHT price beyond that
      list (e.g. a longer pickup, a bigger party, a longer solo ride). Extra
      probes are HARMLESS: the combiner is not required to use every probe, and a
      probe that this reward ignores may be the ONLY thing that reveals a penalty
      in the NEXT reward it has to read. If you find you only probe two or three
      terms, add more -- a shallow probe set is a failed combiner.
Then write `probe_self_check` in the JSON confirming (a) and (b). If you cannot
honestly confirm coverage, you did not finish the task -- go back and add probes.

EXAMPLE EVENT SHAPES (you must produce these or better ones; the exact values
are yours to choose):
  baseline_no_onboard:  picked_up_orders=[], assigned_orders=[p],
                        extra_detour_time=0.0
  dispatch_probe:       picked_up_orders=[], assigned_orders=[p],
                        assigned_dispatch_wait={p:8}, assigned_service_times={p:18},
                        assigned_solo_times={p:5}, assigned_pickup_times={p:5}
  pickup_probe:         picked_up_orders=[], assigned_orders=[p],
                        assigned_solo_times={p:5}, assigned_pickup_times={p:1},
                        assigned_service_times={p:14}   # short pickup vs
  pickup_probe_long:    assigned_solo_times={p:5}, assigned_pickup_times={p:12},
                        assigned_service_times={p:25}   # long pickup
  bundling_detour:      picked_up_orders=[q], assigned_orders=[p],
                        extra_detour_time=2.0
  bundling_no_detour:   picked_up_orders=[q], assigned_orders=[p],
                        extra_detour_time=0.0
  completion_probe:     completed_orders=[p]
  seating_probe:        assigned_party_sizes={p:2}

SELF-TEST (recommended):
  After building your probes, verify they distinguish structurally different
  reward functions by computing w(with_term) - w(without_term) for each term.
  If two terms give indistinguishable differences, add or adjust a probe until
  they separate. This is part of the evolution.
"""


INTERPRETABILITY_RULE = """\
INTERPRETABILITY IS A HEADLINE REQUIREMENT. Your explanation must let a domain
expert predict your combiner's behaviour without reading code:
  - `strategy`: one sentence naming the overall objective-to-skill policy.
  - `description`: 3-5 sentences on the driver states you distinguish, the skill
    each state tends to get, and HOW a revenue-leaning vs a service-leaning episode
    objective shifts the choices. Explain the DISPATCH BEHAVIOUR and the trade-off,
    not the syntax. "returns revenue skill when the objective likes revenue" is NOT
    enough -- say WHY and for WHICH drivers.
"""


CROSSOVER_TASK = """\
Design a CHILD combiner by RECOMBINING the two parent programs below. Both parents
survived selection, and each is stronger on a DIFFERENT set of objective families
(their per-family advantages are printed with them). Your job is not to pick a
winner and tweak it: read what each parent actually does WELL, and build one
program that keeps both strengths.

Concretely:
  - Name, for yourself, the mechanism in parent A that wins A's strong families and
    the mechanism in parent B that wins B's strong families (they are usually
    different probes, different driver-state branches, or different thresholds).
  - Write ONE ``skill_scores`` that routes to the A-mechanism on the situations A
    is good at and the B-mechanism on the ones B is good at -- decided by the
    objective ``w`` and the driver's own state, never by a hard-coded scene.
  - Where the parents disagree on the SAME situation, keep the one whose family
    advantage is higher there, or blend them; do not average blindly.
  - You may add a small improvement of your own, but the child must visibly
    inherit from both parents. A child that is a copy of one parent wastes the
    round.
In ``description``, say which behaviour came from which parent and why.

If either parent uses a probe-event strategy to read ``w``, the child should
refine that probe set too -- probe events are part of the evolving policy.
"""


def _parent_block(label: str, parent: Dict) -> str:
    """Render one crossover parent: name, fitness note, and its code."""
    head = f"# PARENT {label}: {parent.get('name', '?')}"
    strategy = str(parent.get("strategy", "")).strip()
    note = str(parent.get("fitness_note", "")).strip()
    body = [head]
    if strategy:
        body.append(f"strategy: {strategy}")
    if note:
        body.append(f"score: {note}")
    body.append("```python\n" + str(parent.get("code", "")).rstrip() + "\n```")
    return "\n".join(body)


def build_combiner_prompt(
    env_profile: str,
    frozen_skills: Sequence[Dict],
    *,
    current_code: Optional[str] = None,
    current_fitness: Optional[float] = None,
    current_fitness_note: Optional[str] = None,
    parents: Optional[Sequence[Dict]] = None,
    scene_variability: bool = False,
    probe_event_evolve: bool = False,
    reward_spec: Optional[str] = None,
    ignore_pref: bool = False,
    repair_feedback: Optional[str] = None,
) -> Dict[str, str]:
    """Build the ``{"system", "user"}`` prompt for evolving the upper combiner.

    Parameters
    ----------
    env_profile :
        Output of :func:`pref_dispatch.llm.encode.encode_env_profile`.
    frozen_skills :
        Cards of the frozen basis skills (name/objective/description[/signature]).
        These are the ONLY legal keys for ``skill_scores``.
    current_code / current_fitness :
        When improving (MUTATION): the parent combiner and its measured fitness, so
        the model rewrites to beat it. ``None`` on the first (propose) generation.
    current_fitness_note :
        Optional breakdown of the parent's fitness (mean advantage / per-family /
        fallback / objective blindness) shown next to it, so the model can see WHY
        it lost.
    parents :
        v6 CROSSOVER operator: two-or-more parent dicts
        ``{name, strategy, code, fitness_note}``. When given, the task switches to
        :data:`CROSSOVER_TASK` -- recombine the parents' complementary strengths
        into one child -- and ``current_code`` is ignored. The (mu+lambda) loop
        alternates mutation and crossover so a family specialist's mechanism can
        reach a strong all-rounder instead of dying with it.
    scene_variability :
        v2: when True, inject :data:`SCENE_VARIABILITY_MANDATE` so the model writes
        a SCALE-INVARIANT combiner that reads fleet/capacity/demand relatively and
        transfers zero-shot to a new scene.
    reward_spec :
        When given (rendered by :func:`pref_dispatch.llm.reward_spec.reward_spec`),
        inject a ``# EPISODE OBJECTIVE`` section describing the concrete reward the
        objective ``w`` embodies, and switch the output contract to
        :data:`REWARD_OUTPUT_CONTRACT` (which prepends a mandatory
        ``reward_understanding`` CoT field). ``None`` describes ``w`` abstractly.
    ignore_pref :
        Retained for call-site compatibility. In the final-version redesign the
        combiner reads the episode objective ``w`` directly and there is no runtime
        preference dial, so this flag no longer changes the contract; it only
        nudges the task wording toward "one fixed objective" when True.
    repair_feedback :
        If the previous attempt failed sandbox/validation, its error message.
    """
    parts: List[str] = []

    if parents and len(parents) >= 2:
        parts.append("# TASK\n" + CROSSOVER_TASK)
        for label, p in zip("ABCDEF", parents):
            parts.append(_parent_block(label, p))
    elif current_code is None:
        parts.append(
            "# TASK\nDesign the UPPER combiner: one function that scores the frozen "
            "skills for each driver so the resulting blend serves the EPISODE OBJECTIVE "
            "handed to it -- zero-retrain, for objectives it has never seen."
        )
    else:
        parts.append(
            "# TASK\nImprove the UPPER combiner. Rewrite skill_scores to earn MORE "
            "cumulative value under the episode objective (below), keeping the same "
            "contract. It must still serve unseen objectives with no retraining. "
            "If you use probe events to read w, refining them too is part of "
            "improving the policy -- probe event design is not frozen."
        )
        if current_fitness is not None:
            note = ""
            if current_fitness_note:
                note = f"  [breakdown: {current_fitness_note}]"
            parts.append(
                "# PARENT COMBINER (measured fitness = "
                f"{current_fitness:.4g}{note})\n```python\n{current_code}\n```"
            )

    parts.append("# ENVIRONMENT PROFILE\n" + env_profile)
    if reward_spec:
        parts.append(
            "# EPISODE OBJECTIVE (the reward `w` embodies -- read it, then compose "
            "FOR it)\n" + reward_spec
        )
    parts.append("# FROZEN SKILL BASIS\n" + _skill_cards_block(frozen_skills))
    parts.append("# OBJECTIVE (how `w` reaches you)\n"
                 + OBJECTIVE_SPEC + "\n" + OBJECTIVE_DIVERSITY_SPEC)
    parts.append("# FUNCTION CONTRACT\n" + OBJECTIVE_CONTRACT)
    if scene_variability:
        parts.append("# SCENE VARIABILITY\n" + SCENE_VARIABILITY_MANDATE)
    if probe_event_evolve:
        parts.append("# PROBE EVENT EVOLUTION\n" + PROBE_EVENT_EVOLVE_SPEC)
    parts.append("# SANDBOX RULES\n" + SANDBOX_RULES)
    parts.append("# SEED EXAMPLE\n" + FEWSHOT_COMBINER)

    if repair_feedback:
        parts.append(
            "# YOUR PREVIOUS ATTEMPT FAILED -- FIX THIS AND RESUBMIT\n"
            f"{repair_feedback}\n"
            "Return a corrected JSON object addressing exactly this error."
        )

    parts.append("# INTERPRETABILITY\n" + INTERPRETABILITY_RULE)
    contract = REWARD_OUTPUT_CONTRACT if reward_spec else OUTPUT_CONTRACT
    parts.append("# OUTPUT FORMAT\n" + contract)

    return {"system": SYSTEM_PROMPT, "user": "\n\n".join(parts)}


# Fields the extractor must find non-empty (interpretability enforcement).
REQUIRED_EXPLANATION_FIELDS = ("strategy", "description")
# All required fields (explanation + code) for a well-formed combiner response.
REQUIRED_FIELDS = ("combiner_name", "strategy", "description", "code")

# Reward-conditioned variants: add the ``reward_understanding`` CoT gate on top of
# the base contract. Used when ``reward_spec`` is supplied to build_combiner_prompt.
REWARD_REQUIRED_EXPLANATION_FIELDS = ("reward_understanding", "strategy", "description")
REWARD_REQUIRED_FIELDS = ("reward_understanding", "combiner_name", "strategy",
                          "description", "code")
