"""Shared prompt building blocks for skill / combiner evolution.

Everything here is *contract*: the fixed function signatures, the ``phi`` field
list, the real metric menu, the seed few-shot, the safety rules, and -- most
importantly for this project -- the **natural-language explanation requirement**.
Interpretability is a headline selling point, so every generated artefact must
ship human-readable rationale fields, and the prompts say so loudly and show it
in the output schema.

The seed skills and reusable primitives are embedded via ``inspect.getsource``
so the few-shot can never drift from the code that actually runs (§3.4).
"""

from __future__ import annotations

import inspect

from pref_dispatch import skills as _skills_mod

# --------------------------------------------------------------------------- #
# The two-layer context objects. SHARED by every layer that receives them --    #
# skills (below), the combiner, and the repositioner -- so the field list can   #
# never drift between prompts. Anything added to EpisodeStats / GlobalStats     #
# gets described HERE, once.                                                    #
# --------------------------------------------------------------------------- #
PHI_FIELDS_SPEC = """\
  phi_ep: episode-STATIC context (same object every step of the episode).
         An OBJECT, not a dict -- read with attributes. Attributes:
    dist(a, b) -> travel time in MINUTES between two (lon,lat) points (real road
                  net; the network is fixed for the whole episode, so it lives here).
    scale (minutes) -- a leak-free static map scale, safe fallback when the live
                  pending pool is empty.
    num_drivers, driver_capacity, speed_kmh -- fixed fleet/scenario descriptors.
    region_centres -- tuple of (lon,lat), one per region, indexed by region id.
    region_neighbours -- tuple of tuples: region_neighbours[i] lists the region
                  ids adjacent to region i (i itself is NOT in that list).
    od_count -- previous-hour origin-destination matrix as nested tuples:
                  od_count[i][j] is the SHARE of that hour's orders that ran from
                  region i to region j. The WHOLE matrix sums to 1.0 (it is a
                  distribution over flows, not one distribution per row).
    od_out[i] / od_in[i] -- its row / column sums: that hour's share of orders
                  STARTING / ENDING in region i.
    od_orders -- raw order count of that hour, so `share * phi_ep.od_orders`
                  gives an absolute volume. It is 0 when no previous hour was
                  available, and then the whole matrix is zeros.
  phi_step: LIVE per-step context (recomputed every step).
         An OBJECT, not a dict -- read with attributes. Attributes:
    time, num_pending, num_idle, total_free_capacity,
    demand_pressure (= pending / free-capacity),
    mean_solo_time (minutes; the LIVE SCALE -- use this as the unit; fall back to
                    phi_ep.scale only when it is ~0 because no orders are pending).
    region_demand[i] / region_supply[i] -- LIVE per-region pending-order count and
                  idle-driver count, indexed by the same region ids.

  REGION INDICES ARE ONE CONSISTENT SET. `order['origin_region']`,
  `driver_obs['self']['current_region']`, row/column i of `phi_ep.od_count`,
  `phi_ep.region_centres[i]`, `phi_ep.region_neighbours[i]`, `phi_step.region_demand[i]`
  and `phi_step.region_supply[i]` all refer to the same patch of map (nearest
  region centre). So `phi_ep.od_count[order['origin_region']][order['destination_region']]`
  is a legal read and means "how common was this order's flow last hour".
  The OD matrix is measured on the hour BEFORE this episode, never the current
  one, so it is a leak-free prior and NOT a forecast of the orders you are seeing.
  It is not a travel-TIME matrix -- for travel time use `phi_ep.dist(a, b)`.
  GUARD THE OPTIONAL ONES: `region_centres`, `region_neighbours`, `od_count`,
  `od_out`, `od_in`, `region_demand`, `region_supply` are EMPTY tuples when the
  environment has no region layout, and every region id is -1 in that case; with
  a layout but no previous hour on record, `od_orders` is 0 and the OD matrix is
  all zeros. Test `if phi_ep.od_orders:` before leaning on OD numbers and check a
  region id is `>= 0` before indexing, or your function raises and is lost.
  DICTS vs OBJECTS -- getting this wrong raises AttributeError and loses the
  function: `driver_obs` and `order` are DICTS (use `d['key']` / `d.get(k, dflt)`);
  `phi_ep` and `phi_step` are OBJECTS (use `phi_ep.scale`, `phi_step.mean_solo_time`).
  `phi_step.get('mean_solo_time', 0)` does NOT work -- these objects have no
  `.get`. Every attribute listed above is always present, so no defensive read is
  needed. Access `dist` as `phi_ep.dist`.\
"""

# --------------------------------------------------------------------------- #
# Fixed function signatures (the LLM fills only the body).                     #
# --------------------------------------------------------------------------- #
SIGNATURE_SPEC = f"""\
Your skill MUST define this exact function (do not change the signature):

    def score(driver_obs, order, phi_ep, phi_step) -> float:
        # marginal value of inserting `order` into this driver's current route.
        # higher = more preferred. Return -1e9 for an infeasible order.

You MAY also define (optional; a constant 0.0 no-op is used if you omit it):

    def noop_score(driver_obs, phi_ep, phi_step) -> float:
        # value of this driver bidding on NOTHING this step (the wait/idle floor).

Argument contract (read-only; never mutate them):
  driver_obs["self"]: dict with fields
    location=(lon,lat), current_region(int), status(str), capacity(int),
    committed_passengers(int),
    assigned_order_details=[{{order_id, origin=(lon,lat), destination=(lon,lat),
                             num_passengers, onboard(bool), eta(minutes)}}]
  order: dict with fields
    order_id, origin=(lon,lat), destination=(lon,lat), origin_region(int),
    destination_region(int), num_passengers, waiting_time(minutes).
{PHI_FIELDS_SPEC}
  Skills do NOT receive the episode objective `w` (they stay
  objective-specialist); only the combiner + repositioner see `w`.
"""

# --------------------------------------------------------------------------- #
# v10: every objective the stack trains and evaluates on is ADDITIVE over the   #
# per-step events. Stated once here and pasted into BOTH objective-authoring    #
# prompts (the English brief proposer and the reward-code author) so the two    #
# halves of the pipeline cannot disagree about what an objective may be. The    #
# combiner / repositioner prompts state the SAME property from the reading      #
# side, as a guarantee they may rely on when probing `w`.                       #
# --------------------------------------------------------------------------- #
LINEAR_OBJECTIVE_RULE = """\
THE OBJECTIVE MUST BE ADDITIVE OVER EVENTS. This is a hard constraint on the
objective itself, not a style note.

A step's value is the SUM over the individual orders/events in that step, each
one priced ON ITS OWN: so much per order assigned, per drop-off completed, per
shared seat, per minute of service, per minute of empty driving, per unit of
detour. Whatever a single order is worth, two of the same kind of order are worth
exactly twice that, and the tenth one in an hour is worth the same as the first.

Inside that rule you still have wide freedom, and this is where the objectives
differ from each other:
  - WHICH terms carry a price at all -- a price of exactly ZERO is how you say
    "pays only for completed drop-offs, nothing for merely being assigned".
  - WHICH SIGN each price has -- an objective is often defined by what it charges
    for (deadheading, waiting, detour) rather than by what it pays for.
  - HOW BIG each price is relative to the others.
  - Different prices for different KINDS of order: a shared-party order may be
    worth more per order than a solo one, a long trip more than a short one.
    That is a different price for a different item, which is fine.

BANNED, because they all make the price depend on HOW MUCH has already happened:
  - progressive / escalating rates ("the third order this step is worth more than
    the first", "a bonus once the car has carried five riders"),
  - saturating or diminishing returns ("the fifth waiting minute hurts far more
    than the first"),
  - thresholds, caps and floors on a running or episode total ("nothing counts
    below 20 trips", "no more than 100 per driver"),
  - ratios, percentages and averages (a denominator that grows with volume makes
    the per-unit rate depend on the volume),
  - powers, logs, exponentials, or min/max taken over an accumulated quantity.

Why: a dispatcher is expected to read an UNSEEN objective at run time by probing
it, and additivity is what makes a probe informative -- measure a term once and
its rate is known everywhere. An objective that changes its own rate cannot be
read that way, so it is out of scope for this system.
"""

# --------------------------------------------------------------------------- #
# The real metric menu (keys/units EXACTLY match EpisodeMetrics.finalize).     #
#                                                                             #
# v10: this block used to end with a section headed "UNITS ARE THE TRAP, AND   #
# IT HAS ALREADY COST US TWO SKILLS", quoting the two fitness formulas whose   #
# maximum was "refuse every order" and telling the model to check that a busy  #
# hour outscores an empty one. Those two formulas are real and are still on    #
# record in pref_dispatch/llm/skill_audit.py's docstring -- but that section   #
# was our post-hoc diagnosis of a previous run, and handing it over means the  #
# model is repeating our answer rather than reading the numbers. What stays is #
# measured (the TYPICAL ranges), mechanical (the fitness is frozen at gen 0),  #
# and procedural (substitute the ranges and check the ordering).               #
# --------------------------------------------------------------------------- #
METRIC_MENU = """\
An episode rollout returns this metrics dict (these EXACT keys; a fitness may
only read from here). Direction = what "better" means for that term. TYPICAL
RANGE is measured over one real hour at the two ends of the fleet range you are
searched on (200 cars .. 1800 cars) -- read them before you combine two terms:

  revenue            float  higher better  -- sum over assigned orders of
                                              solo_time(min) x party size.
                                              (In this FHVHV data party size is
                                              almost always 1, so revenue tracks
                                              total served trip-minutes.)
                                              TYPICAL 36,000 .. 107,000
  service_rate       float  higher better  -- assigned / total_orders, in [0,1].
                                              TYPICAL 0.20 .. 0.98
  completed          int    higher better  -- orders actually delivered.
                                              TYPICAL 1,200 .. 6,200
  assigned           int    higher better  -- orders assigned to a driver.
                                              TYPICAL 1,800 .. 8,400
  mean_service_time  float  lower  better  -- mean end-to-end service time (min).
                                              TYPICAL 11 .. 18
  detour_total       float  lower  better  -- total extra detour time from
                                              pooling (min); the pooling cost.
                                              TYPICAL 10,000 .. 38,000
  income_gini        float  lower  better  -- driver-income inequality, [0,1].
                                              TYPICAL 0.15 .. 0.17
  income_cv          float  lower  better  -- driver-income coeff. of variation.
                                              TYPICAL 0.77 .. 0.94
  income_mean        float  (context)      -- mean per-driver cumulative reward.
                                              TYPICAL 2.3 .. 3.1
  income_min         float  higher better  -- worst-off driver's cumulative
                                              reward. TYPICAL -5.5 .. -4.8, i.e.
                                              NEGATIVE in a normal hour.

The TYPICAL column is there because these terms are on very different scales: a
rate in [0,1], a count in the thousands and a total in tens of thousands of
minutes all live in the same dict, so two coefficients of similar size do NOT
mean the two terms contribute similarly.

MECHANICS. The fitness you write at generation 0 is FIXED for the whole search:
every later variant of this skill is graded by it, it cannot be changed once the
search starts, and the search returns the maximum of THIS formula, whatever that
maximum turns out to be. Before you submit, put the TYPICAL numbers above into
your own formula and check that the ordering it gives over the outcomes you can
imagine is the ordering you intend.
"""

# --------------------------------------------------------------------------- #
# Safety / sandbox rules.                                                      #
# --------------------------------------------------------------------------- #
SANDBOX_RULES = """\
Code rules (enforced by an AST sandbox -- violating them rejects your skill):
  - Pure functions of the given arguments. No import statements of any kind.
  - Allowed globals: math, np (numpy). Allowed builtins: abs, min, max, sum, len,
    float, int, round, sorted, range, enumerate, zip, map, filter, pow, all, any,
    bool, list, dict, tuple, set.
  - No eval/exec/open/getattr and no attribute access beginning with underscore
    (e.g. no `.__class__`). In particular `getattr(...)` is BANNED even for
    defensive reads -- the sandbox rejects the name itself. Read the DICT
    arguments (`driver_obs`, `order`) with `d['key']` or `d.get('key', default)`
    (getattr on a dict raises AttributeError for ordinary keys anyway). Read
    `phi_ep` / `phi_step` with plain attributes -- they are objects and have no
    `.get`; every documented attribute is always present.
  - Never hard-code a distance/time constant: express thresholds in units of
    phi_step.mean_solo_time (the live scale; fall back to phi_ep.scale when it is
    ~0) so the skill is stable across regimes and map scales.
  - score() must always return a finite float (use -1e9 for infeasible orders).
"""


def _seed_source() -> str:
    """Embed the reusable primitives + the three handwritten seed skills.

    Pulled live from ``pref_dispatch.skills`` so the few-shot is always the code
    that actually runs (and is the fair handwritten baseline the ablation beats).
    """
    parts = []
    for prim in (
        _skills_mod._feasible,
        _skills_mod._pickup_time,
        _skills_mod._solo_time,
        _skills_mod._onboard_slack,
    ):
        parts.append(inspect.getsource(prim))
    for cls in (
        _skills_mod.RevenueSkill,
        _skills_mod.ServiceSkill,
        _skills_mod.EnRouteSkill,
    ):
        parts.append(inspect.getsource(cls))
    return "\n\n".join(parts)


PRIMITIVES_NOTE = """\
Reusable helpers already available in your namespace (call them directly, or
write your own; you do NOT need to redefine them). Pass `phi_ep.dist` wherever a
helper wants `dist`:
  _feasible(driver_obs, order) -> bool          # party fits remaining capacity
  _pickup_time(driver_obs, order, dist) -> float # minutes to reach the pickup
  _solo_time(order, dist) -> float               # direct trip time (minutes)
  _onboard_slack(driver_obs) -> float            # smallest onboard-order eta (min),
                                                 # +inf if the car is empty
"""


def few_shot_seeds() -> str:
    """The full seed few-shot block (primitives + three handwritten skills)."""
    return (
        PRIMITIVES_NOTE
        + "\nThe handwritten seed skills below are valid, runnable examples AND "
        "your starting point. Each specialises one objective; study how they use "
        "phi_step.mean_solo_time as the live scale and how noop_score sets a wait "
        "floor:\n\n"
        "```python\n" + _seed_source() + "\n```"
    )


# --------------------------------------------------------------------------- #
# The interpretability requirement -- stated everywhere output is described.   #
# --------------------------------------------------------------------------- #
INTERPRETABILITY_RULE = """\
INTERPRETABILITY IS A HEADLINE REQUIREMENT. Every artefact you output MUST carry
clear, honest natural-language explanations, not just code. Specifically:
  - `objective`: one sentence naming the SINGLE objective this skill specialises
    in and the intuition (e.g. "maximise served trip-minutes by favouring long
    fares even at some pickup cost").
  - `description`: 2-4 sentences explaining HOW the score logic achieves that
    objective -- what it rewards, what it penalises, and WHEN it prefers to wait
    (noop). A domain expert should understand the behaviour without reading code.
  - `fitness_rationale`: 1-2 sentences on why your chosen fitness function
    faithfully measures this objective from the metrics dict.
Explanations that merely restate the code ("returns revenue minus pickup") are
NOT acceptable; explain the DISPATCH BEHAVIOUR and the trade-off it makes.
"""
