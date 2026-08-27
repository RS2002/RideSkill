"""Frozen evolved skill: next_wave_gravity_wells

Objective: Steer idle cars to accept orders whose DROP-OFF lands in a region that was demand-rich last hour but is currently supply-starved, trading a little pickup cost now for correct next-wave positioning.

Mechanism: opportunity-cost mechanism: score = destination's next-wave positioning gravity MINUS the pickup opportunity cost it burns, with a dynamic noop floor that makes idling a real choice unless the drop-off region genuinely out-pulls waiting

For each order the skill builds a 'gravity' for its destination region equal to how demand-rich that region was last hour (od_in) scaled up sharply when the region is currently supply-starved (live region_demand exceeds region_supply), so a drop-off that leaves the car parked in a soon-to-be-hot, under-served cell scores high. From that gravity it SUBTRACTS an opportunity cost: the pickup deadhead the car must eat and the order's own trip time, both in units of live solo-time, because minutes spent fetching or serving are minutes not available for the next wave. Loaded cars additionally lose credit for any marginal detour imposed on onboard riders. The noop floor is not zero: it holds a positioning value for the car's CURRENT region, so an idle car parked in a strong well waits for a well-aimed order rather than accepting a drop into a dead, over-supplied region — it only bids when a candidate destination out-pulls staying put by more than the pickup cost.

Fitness rationale: Completed volume divided by (service time + marginal detour) rewards keeping the fleet productive while penalising the long pickups/detours that come from parking in the wrong region, so only genuine next-wave positioning that keeps future trips cheap maximises it.
Generated in gen 4 (regime=scenarios). Paradigm B: this
runs at ~zero online LLM cost."""

import math
import numpy as np

from pref_dispatch.skills import (
    _feasible, _pickup_time, _solo_time, _onboard_slack,
)


def score(driver_obs, order, phi_ep, phi_step):
    me = driver_obs['self']
    cap = me['capacity']
    if cap < 1:
        return -1e9
    party = order.get('num_passengers', 1)
    committed = me.get('committed_passengers', 0)
    if committed + party > cap:
        return -1e9

    # live scale in minutes; fall back to static map scale when no pending pool
    scale = phi_step.mean_solo_time
    if not scale or scale <= 1e-6:
        scale = phi_ep.scale
    if not scale or scale <= 1e-6:
        scale = 1.0

    loc = me['location']
    o_org = order['origin']
    o_dst = order['destination']

    # --- pickup deadhead + own trip time, in scale units (opportunity cost) ---
    pickup = phi_ep.dist(loc, o_org)
    trip = phi_ep.dist(o_org, o_dst)
    pickup_u = pickup / scale
    trip_u = trip / scale

    # reject absurd pickups outright (a full solo-trip of deadhead is too dear)
    if pickup_u > 3.0:
        return -1e9

    # --- marginal detour imposed on riders already onboard (loaded cars) ---
    marginal_detour_u = 0.0
    details = me.get('assigned_order_details', [])
    if details:
        # inserting a new pickup detours everyone whose drop still lies ahead;
        # approximate the added leg as the extra fetch from current loc via pickup
        via = phi_ep.dist(loc, o_org) + phi_ep.dist(o_org, o_dst)
        direct = phi_ep.dist(loc, o_dst)
        marginal_detour_u = max(0.0, (via - direct)) / scale

    # --- destination positioning gravity -------------------------------------
    dst_r = order.get('destination_region', -1)
    gravity = 0.0
    if dst_r is not None and dst_r >= 0:
        # last-hour demand density ENDING there (od_in) or STARTING there (od_out):
        # a region that both received and originated flow is a live hub next wave.
        base = 0.0
        if phi_ep.od_orders:
            oin = 0.0
            oout = 0.0
            try:
                oin = phi_ep.od_in[dst_r]
                oout = phi_ep.od_out[dst_r]
            except (IndexError, TypeError):
                oin = 0.0
                oout = 0.0
            # weight originations more: next wave's pickups spawn there
            base = 0.55 * oout + 0.45 * oin

        # live supply-starvation multiplier: reward landing where pending demand
        # already outstrips idle supply (the fleet is short there right now).
        starve = 1.0
        try:
            dem = phi_step.region_demand[dst_r]
            sup = phi_step.region_supply[dst_r]
            # +1 smoothing; ratio saturates so we don't chase empty spikes
            ratio = (dem + 0.5) / (sup + 0.5)
            starve = 1.0 + min(2.0, ratio) * 0.75
        except (IndexError, TypeError):
            starve = 1.0

        # scale gravity into scale-unit magnitude so it competes with costs;
        # od shares are tiny fractions -> multiply up by a fixed pull budget.
        gravity = base * starve * 6.0

    # a soft demand-density fallback when there is no OD prior at all
    if gravity <= 0.0 and dst_r is not None and dst_r >= 0:
        try:
            dem = phi_step.region_demand[dst_r]
            sup = phi_step.region_supply[dst_r]
            gravity = min(2.0, (dem + 0.5) / (sup + 0.5)) * 0.5
        except (IndexError, TypeError):
            gravity = 0.0

    # --- opportunity cost assembly -------------------------------------------
    # pickup deadhead is pure waste; own trip time is half-charged (it delivers
    # a rider AND repositions, so it is not pure cost); onboard detour is waste.
    cost = pickup_u + 0.5 * trip_u + 1.5 * marginal_detour_u

    value = gravity - cost

    # small bonus per rider actually served so a well-aimed multi-seat order
    # is preferred over an equally-aimed single when a seat is free
    value += 0.15 * party

    return float(value)


def noop_score(driver_obs, phi_ep, phi_step):
    me = driver_obs['self']
    scale = phi_step.mean_solo_time
    if not scale or scale <= 1e-6:
        scale = phi_ep.scale
    if not scale or scale <= 1e-6:
        scale = 1.0

    # waiting is a real choice: hold the positioning value of the car's CURRENT
    # region so an idle car sitting in a strong well does not cheaply accept a
    # drop into a dead, over-supplied cell. Loaded cars have a lower floor (they
    # should keep moving to complete committed trips).
    cur = me.get('current_region', -1)
    committed = me.get('committed_passengers', 0)

    floor = 0.0
    if cur is not None and cur >= 0:
        base = 0.0
        if phi_ep.od_orders:
            try:
                base = 0.55 * phi_ep.od_out[cur] + 0.45 * phi_ep.od_in[cur]
            except (IndexError, TypeError):
                base = 0.0
        starve = 1.0
        try:
            dem = phi_step.region_demand[cur]
            sup = phi_step.region_supply[cur]
            ratio = (dem + 0.5) / (sup + 0.5)
            starve = 1.0 + min(2.0, ratio) * 0.75
        except (IndexError, TypeError):
            starve = 1.0
        # floor sits BELOW the seed value of a comparable order (factor 0.5) so a
        # genuinely well-aimed order still clears it, but a mediocre one does not
        floor = base * starve * 6.0 * 0.5

    if committed > 0:
        floor *= 0.4

    return float(floor)
