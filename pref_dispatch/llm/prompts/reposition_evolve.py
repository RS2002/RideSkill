"""Prompt builder for the Feature-3 reposition-scorer evolution (Phase 2).

This arm evolves ONE per-region *reposition scorer*: given an IDLE, unmatched
driver, it rates each preset relocation region by how worthwhile it is to cruise
there empty, so idle cars drift toward near-future demand instead of sitting
still. It is the LLM counterpart to the demand-gravity heuristic in
:mod:`pref_dispatch.reposition`.

The scorer is an independent artefact -- it is NOT a method on an order-scoring
skill, and it has no ``score``/``noop_score`` surface at all. It is wrapped in a
:class:`~pref_dispatch.reposition.Repositioner` (the single handle the dispatch
layer holds) and only ever consulted for idle, unmatched drivers.

Design (mirrors :mod:`pref_dispatch.llm.prompts.pure_phase1`, the "fixed reward"
arm): the objective and the fitness are FIXED by the researcher, not
self-authored. The model must first explain, in natural language, what makes a
good reposition target (a chain-of-thought interpretability gate), then write the
scorer. The fitness is the SAME reward-under-w yardstick as Phases 1-2: the episode
reward under the current objective w (fleet-mean cumulative reward, normalised in a
fixed per-scenario reference frame), averaged over a batch of sampled objectives and
fairness strengths. So a good scorer is one whose repositioning raises whatever
objective the dispatcher currently reads, not a fixed service proxy.

BOUNDARY (safety): the model authors ONLY the per-region base attractiveness. The
deterministic coordinated-spreading, the stay rules, and the emission of the
concrete ``{"relocate": region_index}`` action all stay in
:mod:`pref_dispatch.reposition` and are NEVER LLM-authored -- the env's action
contract is never in model hands. So the scorer cannot emit an illegal action; the
worst a bad scorer can do is pick a poor (but in-range) region, which the fitness
then punishes.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from pref_dispatch.llm.prompts.common import INTERPRETABILITY_RULE, SANDBOX_RULES

# The researcher-FIXED objective + fitness the scorer is graded by: the reward under
# the current objective w (same yardstick as Phases 1-2). Kept human-readable because
# the model must reason about it first.
FIXED_REPOSITION_OBJECTIVE = """\
You design a REPOSITION SCORER for idle empty vehicles in a ride-POOLING fleet.
When a driver is idle and won no order this step, the dispatcher may send it to
cruise (empty) toward one preset "relocation region" so it is better placed for
near-future demand. Your scorer rates each candidate region; the dispatcher then
applies its OWN deterministic logic on top (it spreads cars across hot regions so
they do not all flock to one, refuses moves that are too short to bother with, and
emits the actual relocate action). You ONLY provide the per-region base score.

The objective is FIXED (you do NOT design it). A good reposition scorer should:
  - Send idle cars toward regions with genuine near-future PICKUP demand (many
    pending orders originate there, or will), so a car is already close when the
    next order appears -> shorter passenger waiting time and more orders served.
  - Discount a region by how far the driver must cruise to reach it: a distant
    hotspot is worth less than a comparably-hot near one (empty cruising is time
    the car is not earning, and demand it chases may be gone by arrival).
  - Avoid piling cars where free supply is already plentiful: a region thick with
    idle/relocating cars needs no more; prefer under-served demand.
  - Leave a car where it is (score its current region low, or return nothing for a
    region) when no target is clearly better -- needless empty cruising is waste.

You are graded by the SAME fitness the whole system optimises (the objective the
dispatcher already reads): the episode reward under the current objective w. But you are
NOT scored on that reward directly -- you are scored on WHAT YOUR REPOSITIONING IS WORTH.
On every (scene, objective, fairness-strength) cell your score is

    (YOUR episode reward - the episode reward with repositioning switched OFF,
     i.e. every idle car left parked, on the SAME scene and the SAME seed)
    / how much this round's programs disagree about that same difference

and your fitness is the mean over the cells (a spread of objectives and fairness strengths,
so ONE frozen scorer must best-respond across all of them). Two consequences you should
plan around:

  - REWARD SCALES DO NOT MATTER. One objective may pay 10x what another pays; dividing by
    the round's own disagreement removes that, so a cell where the money is big cannot
    outvote a cell where it is small. Do not try to guess which objective is "worth more".
  - THE SIGN IS ABSOLUTE, not a rank. 0.00 means your scorer was worth exactly as much as
    leaving every car parked. NEGATIVE means sending cars around ACTIVELY LOST money
    against doing nothing. +1 means your gain over doing nothing is one whole spread above
    how much the field's gains vary. Beating the other candidates is NOT the target --
    beating "do nothing" is, and every candidate in a round can be negative at once.

Repositioning is worthwhile ONLY insofar as sending idle cars toward near-future demand
lets the fleet serve more of the objective: shorter passenger waits and more (or
better-paying, depending on w) orders completed. Empty cruising that does not pay for
itself LOWERS the reward -- the car is not earning while it repositions and may arrive
after the demand is gone -- and that now shows up literally, as a NEGATIVE score, not
merely as a worse rank. The built-in demand-gravity heuristic is rolled in the same field
as a landmark. You are NOT graded by a fixed service formula, you are graded by the
objective itself.

CRASHING SCORES EXACTLY 0, AND IT IS NOT A HIDING PLACE. If your function raises on a
driver, that car is PARKED for that step -- which is precisely the do-nothing baseline the
score subtracts. So a scorer that breaks everywhere scores 0.00: it inherits nothing from
the heuristic and buys nothing by failing. Anything you actually want to be paid for has
to come from moves that beat parking, and a scorer that breaks OFTEN is repaired once and
then eliminated.

The objective `w` itself VARIES across episodes -- it is drawn from a distribution of
term-different reward families: linear coefficient mixes; COMPLETION-gated rewards that pay
on drop-off; POOLING/seating rewards that pay per filled seat; and rewards that penalise
empty cruising and idling. Every one of them is LINEAR in the per-step event terms (a fixed
price per assignment, per drop-off, per seat, per minute), so the price you read off a probe
is constant and a probe with twice the volume returns twice the value. You may probe `w` on
a small event dict to learn what a FINISHED trip is worth in THIS episode and weigh it
against the cruising cost. The same kappa (region supply/demand) can therefore justify very
different moves under different objectives: under a completion-gated or empty-averse
objective, repositioning must pay for itself in finished trips (cruise only where pickup is
genuinely imminent); under a seating objective, steer toward regions with multi-party
pickups; under a throughput objective, raw proximity to pending demand dominates. Read `w`
AND the live kappa together, every step; a scorer that maximises one fixed service formula
regardless of the objective will be outcompeted.

WHETHER YOU ACTUALLY RESPOND IS MEASURED, NOT ASSUMED. Every round reports back two
numbers on your own run: OBJECTIVE BLINDNESS and FAIRNESS BLINDNESS -- 0 means the set of
regions you sent cars to visibly MOVED when the objective (or the fairness strength)
changed, 1 means it never moved. They are diagnostics, not fitness terms: nothing is
deducted for a 1.00. But a 1.00 is a statement about your program -- it reads `w`
cosmetically (computing something from it and then not letting it change the argmax) or
not at all -- and it is the reliable predictor of a weak family score, which IS charged.
The round also reports DEAD FAMILIES: families where your fleet earned the same episode
reward as everyone else's, meaning no program in that round -- yours included -- sent a
car anywhere different when the objective changed shape. Both readings tell you where the
unclaimed points are: make the TARGET REGION visibly depend on what you are asked to
maximise, not just the score you attach to it.

The FAIRNESS STRENGTH also varies across episodes, and it is the second thing you must
read. Before matching, the dispatcher multiplies every driver's bid by a budget that is
LARGER for drivers who have earned less so far -- `phi_ep.fairness_strength` is how hard it
pushes. At 0 the budget is off and matching is pure efficiency. Around 0.5 it decides close
calls. Above 1 it can hand a good order to a poor driver over a much better-placed rich one.
That changes which car is worth putting where:
  - At strength 0, position is everything: the car nearest the best demand wins it, so cruise
    whichever idle car is closest to the strongest under-served demand.
  - At a high strength, a poor driver wins good orders wherever it is, while a rich driver
    can be parked on top of demand and still lose it. Cruising the RICH car toward a hotspot
    buys little; getting the POOR cars within reach of ANY demand is what turns the budget's
    intent into completed trips instead of orders assigned to cars too far away to serve
    them quickly.
You can see exactly this in the observation: `driver_obs["fairness_budget"]` is THIS driver's
own multiplier for this step and `driver_obs["driver_budgets"]` is the whole fleet's. Above 1
means it has earned below the fleet mean and its bids are being boosted; below 1 means it is
being damped. At strength 0 every multiplier is exactly 1.0, which is the honest signal that
the axis is off. A scorer that ignores `phi_ep.fairness_strength` and these multipliers is
answering the wrong question in half the episodes it is graded on, and its weakest strength
band is charged directly in selection.
"""

# The function contract for a reposition scorer. It defines exactly ONE function:
# reposition_scores. There is deliberately NO score/noop_score -- a reposition
# scorer is not an order-scoring skill and is never bid into an order match.
REPOSITION_SIGNATURE_SPEC = """\
Your scorer MUST define exactly this function (do not change the signature):

    def reposition_scores(driver_obs, phi_ep, phi_step, kappa, w) -> dict:
        # {region_index: base_score} over a SUBSET of the preset regions.
        # Higher = more worth cruising to. Return {} to defer entirely to the
        # built-in demand-gravity heuristic (no opinion). Every key MUST be an int
        # region index in range(len(driver_obs["relocation_points"])); an
        # out-of-range key is rejected. Values must be finite numbers.
        # The dispatcher considers the driver's CURRENT region, its immediate
        # neighbours, AND the handful of GLOBALLY HOTTEST regions (highest
        # kappa.eff_demand) -- so an idle car can cruise toward a city-wide hotspot
        # even when it is NOT adjacent. Score the current region + its neighbours
        # AND the few regions with the largest kappa.eff_demand; a far hotspot is
        # worth naming because the dispatcher discounts it by cruise time and will
        # only send the car there if the demand beats the empty drive.

Argument contract (read-only; never mutate any of them, including kappa):
  driver_obs["self"]: {location=(lon,lat), current_region(int), status(str),
                       capacity(int), committed_passengers(int)}
  driver_obs["relocation_points"]: tuple of region-centre (lon,lat) coords. Region
                       index r is relocation_points[r]. len(...) = number of regions.
  driver_obs["region_neighbours"]: tuple; region_neighbours[r] = tuple of region
                       indices adjacent to region r (the env's own adjacency graph).
                       The candidate targets are current_region + its neighbours +
                       the few globally hottest regions (top kappa.eff_demand), so a
                       far but genuinely hot region is a valid target too.
  driver_obs["pending_orders"]: list of {order_id, origin=(lon,lat),
                       destination=(lon,lat), origin_region(int),
                       destination_region(int), num_passengers, waiting_time(min)}.
                       These are the unserved pickups = the demand to chase. The
                       region fields use the SAME region indexing as everything
                       else here, so you can bucket the live pool by region
                       directly instead of re-deriving it from coordinates.
  driver_obs["all_drivers"]: {driver_id: {location=(lon,lat), status(str),
                       onboard_passengers(int)}} -- the whole fleet's public state.
                       status in {"idle","relocating","to_pickup","to_dropoff"};
                       "idle"/"relocating" cars are the FREE supply competing with
                       this driver for the same demand.
  driver_obs["fairness_budget"]: float -- THIS driver's fairness multiplier for this
                       step. The matcher multiplies its bids by it before matching.
                       > 1 = earned below the fleet mean, bids boosted (it can win
                       orders from further away); < 1 = earned above, bids damped
                       (parking it on a hotspot buys less). Exactly 1.0 for everyone
                       when phi_ep.fairness_strength is 0.
  driver_obs["driver_budgets"]: {driver_id: multiplier} for the WHOLE fleet, same
                       scale. Use it to see whether the OTHER free cars near a
                       hotspot are boosted or damped -- cruising into demand that a
                       boosted rival will win anyway is wasted empty distance.
  phi_ep: episode-STATIC context (same object every step). Fields:
      dist(a, b) -> travel time in MINUTES between two (lon,lat) points (real roads;
                    the network is fixed for the episode, so dist lives here).
      scale (minutes) -- a leak-free static map scale (fallback unit).
      region_centres -- the canonical region layout; num_drivers, driver_capacity.
      region_neighbours -- same adjacency graph as driver_obs["region_neighbours"].
      od_count / od_out / od_in / od_orders -- the PREVIOUS hour's origin-destination
                    matrix over those same regions, as nested tuples (not numpy).
                    od_count[i][j] is the share of that hour's orders that ran from
                    region i to region j; the WHOLE matrix sums to 1.0. od_out[i] and
                    od_in[i] are its row / column sums, i.e. the share of orders that
                    STARTED / ENDED in region i. od_orders is that hour's raw order
                    count, so `share * od_orders` is an absolute volume; it is 0 when
                    no previous hour was on record and then the matrix is all zeros
                    (guard with `if phi_ep.od_orders:`).
                    THIS IS THE STRUCTURAL DEMAND PRIOR A REPOSITIONER WANTS: kappa
                    tells you where demand is RIGHT NOW, od_in/od_out tell you where
                    it habitually appears and where trips habitually END -- so a car
                    sent into region j on od_count[i][j] arrives where the next wave
                    of drop-offs and pickups tends to be. It is measured on the hour
                    BEFORE this episode, never the current one, so it is a leak-free
                    prior and NOT a forecast of the orders you can see.
      fairness_strength (float, >= 0, uncapped) -- how hard the wage-equalising
                    budget above pushes this episode. 0 = off (all multipliers 1.0);
                    <= 1 = it re-orders close calls; > 1 = it can override a large
                    score gap outright. Episode-static: one rollout, one strength.
  phi_step: LIVE per-step context. Fields:
      time, num_pending, num_idle, total_free_capacity, demand_pressure,
      mean_solo_time (minutes; the LIVE SCALE -- use it as the unit, fall back to
      phi_ep.scale when ~0),
      region_demand[r] / region_supply[r] -- the same per-region live counts kappa
      is built from, as plain tuples. Prefer `kappa` below: it is netted and decayed
      as this step's earlier cars are dispatched, while these are the raw snapshot.
  kappa: the shared per-region demand/supply state (already netted for you). Read
      (do NOT mutate) these numpy arrays, indexed by region:
        kappa.demand[r]      -- raw pending demand (party sizes) in region r.
        kappa.supply[r]      -- free (idle/relocating) cars nearest region r.
        kappa.eff_demand[r]  -- demand netted of supply and decayed as earlier idle
                                cars are already sent this step (the live signal to
                                chase; the dispatcher updates it sequentially).
  w: the episode OBJECTIVE -- an LLM-authored callable reward `w(event) -> float`,
     or None when objective-blind. You MAY call it on a probe event to self-derive
     what a finished trip is worth in this episode, but you MUST also work when w
     is None. An `event` dict to probe with has these keys: assigned_orders(list),
     assigned_party_sizes(dict), assigned_solo_times(dict),
     assigned_service_times(dict), completed_orders(list), picked_up_orders(list),
     distance_moved(float), time_moved(float), is_empty_move(bool),
     is_idle_wait(bool), extra_detour_time(float).

REGION INDICES ARE ONE CONSISTENT SET. Region r means the same patch of map in
driver_obs["relocation_points"], driver_obs["region_neighbours"], the driver's
current_region, an order's origin_region / destination_region, kappa.demand[r] /
kappa.supply[r] / kappa.eff_demand[r], phi_step.region_demand[r], and row/column r
of phi_ep.od_count. They are all bucketed by the same nearest-region-centre rule,
so mixing them in one expression is safe. A region id is -1 only when the
environment has no layout at all.

Express any distance/time threshold in units of phi_step.mean_solo_time (fall back
to phi_ep.scale) so the scorer is stable across map scales and demand regimes
(never hard-code a minutes/lon-lat constant).
"""

OUTPUT_CONTRACT = """\
Respond with ONE JSON object and nothing else (no prose before/after, no markdown
outside the JSON). Schema:

{
  "reposition_understanding": "<2-4 sentences, FIRST: in your own words, what makes
                 a region worth cruising an idle empty car toward, and what a
                 service-maximising, waste-avoiding reposition scorer must therefore
                 do. This is a chain-of-thought gate: reason before you write.>",
  "skill_name":  "<short snake_case id, e.g. demand_gravity_scorer>",
  "objective":   "<ONE sentence: the per-region scoring policy you will use>",
  "objective_read_check": "<2-3 sentences: HOW your scorer makes the TARGET REGION
                 depend on the objective `w` (not just the score magnitude). Name at
                 least one objective family you respond to and the concrete
                 mechanism -- e.g. a completion-gated w pushes you to cruise only to
                 regions with genuinely imminent pickups, an empty-averse w pushes
                 you away from long empty cruises, a seating w pushes you toward
                 multi-party regions, a length-driven w pushes you toward long-fare
                 origins. If the described mechanism does not change WHICH region
                 wins the argmax when `w` changes, it is cosmetic and will be caught
                 by the objective-blindness report.>",
  "description": "<2-4 sentences: HOW your scores rank regions -- how you weigh
                 demand, cruise distance, and existing supply, and when you return
                 {} / a low score to keep a car put. Explain behaviour, not code.>",
  "code": "def reposition_scores(driver_obs, phi_ep, phi_step, kappa, w):\\n    ..."
}

There is NO fitness field: the objective and fitness are fixed and given above; you
only write the scoring policy that maximises the fixed fitness.
"""

SYSTEM_PROMPT = """\
You are an expert in ride-POOLING fleet operations and empty-vehicle
repositioning. You are given a FIXED objective (send idle empty cars toward
near-future demand without wasteful cruising) and a FIXED fitness (the service
improvement it brings over not repositioning). You must first explain, in natural
language, what makes a good reposition target, and then write ONE small, robust,
interpretable Python scorer that rates each preset region for an idle driver. You
always answer with exactly one JSON object matching the requested schema.
"""


CROSSOVER_TASK = """\
Design a CHILD reposition scorer by RECOMBINING the two parent programs below. Both
parents survived selection, and each is stronger on a DIFFERENT set of cells -- the
per-objective-family AND per-fairness-strength advantages are printed with them. Your
job is not to pick a winner and tweak it: read what each parent actually does WELL,
and build one program that keeps both strengths.

Concretely:
  - Name, for yourself, the mechanism in parent A that wins A's strong cells and the
    mechanism in parent B that wins B's strong cells. Usually they differ in how they
    weigh demand against cruise time, in which regions they will even consider, or in
    how they react to `phi_ep.fairness_strength` and the per-driver multipliers.
  - Write ONE ``reposition_scores`` that uses the A-mechanism on the situations A is
    good at and the B-mechanism on the ones B is good at -- decided by the objective
    `w`, the live kappa, and the fairness strength, never by a hard-coded scene.
  - Where the parents disagree about the SAME situation, keep the one whose advantage
    is higher there, or blend them; do not average blindly.
  - You may add a small improvement of your own, but the child must visibly inherit
    from both parents. A child that is a copy of one parent wastes the round.
In ``description``, say which behaviour came from which parent and why.
"""


def _parent_block(label: str, parent: Dict) -> str:
    """Render one crossover parent: name, objective, score breakdown, and its code."""
    body = [f"# PARENT {label}: {parent.get('name', '?')}"]
    objective = str(parent.get("objective", "")).strip()
    note = str(parent.get("fitness_note", "")).strip()
    if objective:
        body.append(f"objective: {objective}")
    if note:
        body.append(f"score: {note}")
    body.append("```python\n" + str(parent.get("code", "")).rstrip() + "\n```")
    return "\n".join(body)


def build_reposition_prompt(
    env_profile: str,
    *,
    current_code: Optional[str] = None,
    current_fitness: Optional[float] = None,
    current_fitness_note: Optional[str] = None,
    parents: Optional[Sequence[Dict]] = None,
    repair_feedback: Optional[str] = None,
) -> Dict[str, str]:
    """Build the ``{"system","user"}`` prompt for the reposition-scorer arm.

    Parameters
    ----------
    env_profile :
        Output of :func:`pref_dispatch.llm.encode.encode_env_profile`.
    current_code / current_fitness :
        When improving (MUTATION): the parent scorer and its measured fitness, so the
        model rewrites to beat it. ``None`` on the first (propose) generation.
    current_fitness_note :
        Optional breakdown of that fitness (mean advantage / per-family /
        per-strength-band / fallback rate) shown next to it, so the model can see
        WHERE it lost rather than only that it lost.
    parents :
        CROSSOVER operator, identical in spirit to Phase 2's: two-or-more parent
        dicts ``{name, objective, code, fitness_note}``. When given the task switches
        to :data:`CROSSOVER_TASK` and ``current_code`` is ignored, so a scorer that is
        excellent only at high fairness strength can pass its mechanism to a strong
        all-rounder instead of dying with it.
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
            "# TASK\nWrite ONE reposition scorer that MAXIMISES the fixed fitness "
            "below. First explain what makes a good reposition target, then write "
            "the scorer. It rates preset regions for an idle empty car; the "
            "dispatcher applies spreading + stay rules + the actual move on top."
        )
    else:
        parts.append(
            "# TASK\nImprove your reposition scorer to raise the FIXED fitness "
            "(more service, less waiting, less wasteful cruising), keeping the same "
            "contract. Re-explain the (improved) behaviour."
        )
        if current_fitness is not None:
            note = f"  [breakdown: {current_fitness_note}]" if current_fitness_note else ""
            parts.append(
                "# PARENT SCORER (measured fitness = "
                f"{current_fitness:.4g}{note})\n```python\n{current_code}\n```"
            )

    parts.append("# FIXED OBJECTIVE & FITNESS (given -- you do NOT change it)\n"
                 + FIXED_REPOSITION_OBJECTIVE)
    parts.append("# ENVIRONMENT PROFILE\n" + env_profile)
    parts.append("# FUNCTION CONTRACT\n" + REPOSITION_SIGNATURE_SPEC)
    parts.append("# SANDBOX RULES\n" + SANDBOX_RULES)

    if repair_feedback:
        parts.append(
            "# YOUR PREVIOUS ATTEMPT FAILED -- FIX THIS AND RESUBMIT\n"
            f"{repair_feedback}\n"
            "Return a corrected JSON object addressing exactly this error."
        )

    parts.append(
        "# SELF-CHECK BEFORE YOU SUBMIT (do this EVERY time you write a scorer)\n"
        "The objective `w` is handed to you as a live callable, and whether you "
        "actually respond to it is MEASURED (objective blindness: 0 = the set of "
        "regions you send cars to moved when `w` changed, 1 = it never moved). "
        "Before submitting, confirm BOTH:\n"
        "  (a) NOT COSMETIC -- you read `w` and the value you compute from it "
        "actually enters THE ARGMAX, i.e. a different shape of `w` changes WHICH "
        "region wins, not just the score you attach to it. If it never changes the "
        "winning region, you are objective-blind on that family and its cell is "
        "unanswerable.\n"
        "  (b) FAMILY COVERAGE -- the mechanism in `objective_read_check` names at "
        "least one objective family you respond to AND how (completion-gated -> only "
        "cruise where pickup is imminent; empty/idle-averse -> discount long empty "
        "cruises; seating -> multi-party regions; length-driven -> long-fare "
        "origins; throughput -> raw live demand proximity). A scorer that only reads "
        "`w`, multiplies by a constant, and returns the same rankings is FAILED -- "
        "the round reports your blindness directly.\n"
        "Also re-confirm fairness: at a high `phi_ep.fairness_strength` a rich car "
        "parked on a hotspot is worth less than a poor car brought anywhere near "
        "demand; a scorer that ignores the per-driver multipliers is blind on the "
        "strength axis, which selection charges directly."
    )

    parts.append("# INTERPRETABILITY\n" + INTERPRETABILITY_RULE)
    parts.append("# OUTPUT FORMAT\n" + OUTPUT_CONTRACT)

    return {"system": SYSTEM_PROMPT, "user": "\n\n".join(parts)}


# Interpretability fields the extractor must find non-empty. ``reposition_understanding``
# is the CoT gate specific to this arm (the model must reason about targets first).
REQUIRED_EXPLANATION_FIELDS = ("reposition_understanding", "objective",
                               "objective_read_check", "description")
# All required fields (NO fitness_code -- the fitness is fixed and researcher-given).
REQUIRED_FIELDS = ("reposition_understanding", "skill_name", "objective",
                   "objective_read_check", "description", "code")
