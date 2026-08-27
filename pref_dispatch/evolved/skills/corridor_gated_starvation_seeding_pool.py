"""Frozen evolved skill: corridor_gated_starvation_seeding_pool

Objective: Maximise riders served per minute of added deadhead by inserting each order into whichever car's current route already points down the same corridor, so pooling adds passengers with near-zero marginal detour.

Mechanism: hard corridor-alignment gate then riders-per-marginal-detour ranking, with a starvation-adaptive reach ONLY on empty-car corridor seeding

For a loaded car it hard-rejects any order whose direction, pickup or fetch fails a strict corridor test, then ranks survivors by party-size divided by marginal detour so near-free on-corridor riders win. For an empty car it seeds a new corridor, and here alone it stretches the acceptable pickup reach when the order's origin region has more pending demand than idle supply, and lifts the seed value there — pulling idle cars toward starved corridors so thin fleets serve more riders without ever relaxing the pooling gate that keeps deadhead near zero. It prefers to wait (high noop) when a car is already loaded and its onboard riders' slack is tight, so only a genuinely cheap aligned insert clears the floor.

Fitness rationale: It multiplies completed-rider volume by a detour-per-rider efficiency and damps below a service-rate floor, so the score rises only when MORE riders are pooled at genuinely low marginal deadhead — exactly the served-per-added-deadhead objective, and it rewards the starved-region seeding only insofar as those reached riders end up cheap.
Generated in gen 4 (regime=scenarios). Paradigm B: this
runs at ~zero online LLM cost."""

import math
import numpy as np

from pref_dispatch.skills import (
    _feasible, _pickup_time, _solo_time, _onboard_slack,
)


def score(driver_obs, order, phi_ep, phi_step):
    me = driver_obs['self']
    cap = max(int(me['capacity']), 1)
    party = max(1, int(order.get('num_passengers', 1)))
    committed = int(me.get('committed_passengers', 0))
    # capacity feasibility: never overfill.
    if committed + party > cap:
        return -1e9

    # live scale unit (minutes); fall back to static map scale.
    unit = phi_step.mean_solo_time
    if not unit or unit <= 1e-6:
        unit = phi_ep.scale
    if not unit or unit <= 1e-6:
        unit = 1.0

    dist = phi_ep.dist
    loc = me['location']
    o_org = order['origin']
    o_dst = order['destination']
    pickup = dist(loc, o_org)
    ride = dist(o_org, o_dst)

    details = me.get('assigned_order_details', []) or []
    status = me.get('status', 'idle')
    have_route = (status != 'idle') and (committed > 0 or len(details) > 0)

    oreg = order.get('origin_region', -1)
    dreg = order.get('destination_region', -1)

    # OD-corridor prior (busy same-hour flow), guarded.
    corridor = 0.0
    if phi_ep.od_orders and oreg is not None and oreg >= 0 and dreg is not None and dreg >= 0 and oreg < len(phi_ep.od_count):
        row = phi_ep.od_count[oreg]
        if dreg < len(row):
            corridor = float(row[dreg])

    def vec(a, b):
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        m = math.hypot(dx, dy)
        if m <= 1e-12:
            return (0.0, 0.0, 0.0)
        return (dx / m, dy / m, m)

    # ---- Local starvation ONLY for empty-car seeding: origin region with more
    # pending demand than idle supply -> reach further to pull idle cars in. ----
    local_slack = 0.0
    if oreg is not None and oreg >= 0 and len(phi_step.region_supply) > oreg:
        sup = float(phi_step.region_supply[oreg])
        dem = float(phi_step.region_demand[oreg]) if len(phi_step.region_demand) > oreg else 0.0
        local_slack = min(1.0, dem / (1.0 + sup + dem))

    # ---- EMPTY car: seed a corridor. Strict base reach, stretched in a starved
    # region so thin fleets close the gap to under-served demand. ----
    if not have_route:
        reach = (2.5 + 1.0 * local_slack) * unit
        if pickup > reach:
            return -1e9
        p = pickup / unit
        r = ride / unit
        base = 0.6 + 0.4 * min(r, 1.5) - 0.5 * p + 8.0 * corridor
        # lift seed value toward starved corridors (raises weak-scale volume).
        base = base + 1.2 * local_slack
        return float(base)

    # ---- LOADED car: spine = furthest committed drop-off. STRICT gates (Parent 2). ----
    anchor = None
    best_far = -1.0
    for det in details:
        dpp = det.get('destination')
        if dpp is None:
            continue
        f = dist(loc, dpp)
        if f > best_far:
            best_far = f
            anchor = dpp
    if anchor is None:
        if pickup > 2.5 * unit:
            return -1e9
        return 0.5

    rvx, rvy, rmag = vec(loc, anchor)
    ovx, ovy, omag = vec(o_org, o_dst)
    if rmag <= 1e-12 or omag <= 1e-12:
        return -1e9

    # HARD GATE 1: order must travel roughly the SAME way as the spine (STRICT).
    cos_ride = rvx * ovx + rvy * ovy
    if cos_ride < 0.34:
        return -1e9
    # HARD GATE 2: pickup must not send the car backwards unless trivially short.
    pvx, pvy, pmag = vec(loc, o_org)
    cos_pick = (rvx * pvx + rvy * pvy) if pmag > 1e-12 else 1.0
    if cos_pick < 0.0 and pickup > 0.35 * unit:
        return -1e9
    # HARD GATE 3: fetch cost stays small relative to trip-scale (STRICT).
    if pickup > 1.2 * unit:
        return -1e9

    # ---- RANK survivors by counterfactual riders / marginal-detour. ----
    spine = best_far / unit
    via = (dist(loc, o_org) + dist(o_org, anchor)) / unit
    detour_pick = max(0.0, via - spine)
    via_drop = (dist(o_org, o_dst) + dist(o_dst, anchor)) / unit
    direct_drop = dist(o_org, anchor) / unit
    detour_drop = max(0.0, via_drop - direct_drop)
    marginal_detour = detour_pick + detour_drop

    slack = 0.0
    for det in details:
        if det.get('onboard'):
            eta = det.get('eta')
            if eta is not None:
                slack += float(eta)
    slack = slack / unit
    tol = 1.0 + max(0.0, slack)

    if marginal_detour > tol:
        return -0.5 * (marginal_detour - tol)

    efficiency = float(party) / (0.3 + marginal_detour)
    align = 0.5 * (cos_ride + 1.0)
    align = align * align

    return float(efficiency * (0.6 + 0.4 * align) + 3.0 * corridor - 0.15 * (pickup / unit))


def noop_score(driver_obs, phi_ep, phi_step):
    me = driver_obs['self']
    unit = phi_step.mean_solo_time
    if not unit or unit <= 1e-6:
        unit = phi_ep.scale
    if not unit or unit <= 1e-6:
        unit = 1.0
    committed = int(me.get('committed_passengers', 0))
    details = me.get('assigned_order_details', []) or []
    have_route = (me.get('status', 'idle') != 'idle') and (committed > 0 or len(details) > 0)
    if not have_route:
        # empty car: low floor so it readily seeds a corridor; lower it further
        # when the fleet is starved so idle cars reach out to unmatched demand.
        dp = phi_step.demand_pressure
        if dp is None or dp < 0:
            dp = 0.0
        starve_relief = 0.15 / (1.0 + 4.0 * dp)
        return float(0.4 - starve_relief)
    cap = max(int(me['capacity']), 1)
    load_frac = min(committed / cap, 1.0)
    slack = 0.0
    for det in details:
        if det.get('onboard'):
            eta = det.get('eta')
            if eta is not None:
                slack += float(eta)
    slack = slack / unit
    return float(1.0 + 1.5 * load_frac + 2.0 / (1.0 + max(0.0, slack)))
