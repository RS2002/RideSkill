"""Frozen evolved skill: waiting_rank_detour_seeker

Objective: Rescue the oldest, longest-waiting orders that efficiency-seeking skills have passed over, by deliberately accepting high pickup cost and detour when an order's waiting time signals it has been repeatedly skipped.

Mechanism: two-stage lexicographic: shortlist survivors by an absolute waiting-time desperation floor, then rank the desperate ones by the detour cost they force us to swallow (higher cost = higher score) with a plateau so no single order dominates

The score is dominated by a lexicographic desperation tier: an order whose waiting time exceeds successive multiples of the live solo-time scale jumps into a strictly higher band, so a truly abandoned order always outbids any fresh one regardless of cost. Within a band the skill then ADDS a reward that GROWS with normalised pickup fetch plus ride detour (capped so one huge trip cannot monopolise the fleet), meaning among equally-old orders it deliberately grabs the one that costs the most to reach — exactly the far, high-detour rescue that efficiency skills skip. Onboard riders only mildly discount the bid, never gate it, and demand pressure loosens the desperation thresholds so thin fleets rescue sooner. It prefers to wait (positive noop) only when parked in a region that historically received abandoned demand and is currently supply-starved, holding the car ready to pounce on the next stale order rather than burning itself on a fresh cheap one.

Fitness rationale: The fitness only rewards completed orders when detour-per-order AND mean service time both clear a rescue threshold, so it faithfully measures the objective of accepting real detour/wait sacrifice to serve neglected orders rather than cheap efficient throughput.
Generated in gen 5 (regime=scenarios). Paradigm B: this
runs at ~zero online LLM cost."""

import math
import numpy as np

from pref_dispatch.skills import (
    _feasible, _pickup_time, _solo_time, _onboard_slack,
)


def score(driver_obs, order, phi_ep, phi_step):
    scale = phi_step.mean_solo_time
    if scale <= 0:
        scale = phi_ep.scale
    scale = max(scale, 1e-9)

    selfd = driver_obs['self']
    cap = selfd['capacity']
    party = order['num_passengers']
    if party > (cap - selfd['committed_passengers']):
        return -1e9

    # Demand pressure loosens desperation thresholds: thin/busy fleets rescue sooner.
    dp = phi_step.demand_pressure
    press = dp / (1.0 + dp)          # 0..1
    loosen = 1.0 - 0.35 * press      # lower band edges when busy

    # ---- STAGE 1: LEXICOGRAPHIC DESPERATION TIER --------------------------
    # Waiting time in units of the live solo-time scale. Each threshold crossed
    # promotes the order into a STRICTLY higher band (large integer step), so an
    # abandoned order always outbids a fresh one no matter what it costs to reach.
    wn = order['waiting_time'] / scale
    t1 = 0.75 * loosen
    t2 = 1.75 * loosen
    t3 = 3.00 * loosen
    if wn >= t3:
        band = 3
    elif wn >= t2:
        band = 2
    elif wn >= t1:
        band = 1
    else:
        band = 0
    # Each band is worth 1000 units apart -> lexicographic dominance of older orders.
    tier = 1000.0 * band

    # ---- STAGE 2: RANK BY SACRIFICE (higher cost = higher score) ----------
    # Among equally-desperate orders we DELIBERATELY prefer the one that costs the
    # most to reach + carry, up to a plateau so no single monster trip monopolises
    # the whole fleet. This is the inverse of an efficiency ranker.
    loc = selfd['location']
    fetch = phi_ep.dist(loc, order['origin']) / scale
    ride = phi_ep.dist(order['origin'], order['destination']) / scale
    cost_n = fetch + ride
    # Saturating reward that RISES with cost: 0 at cost 0, plateaus ~1 far out.
    sacrifice = cost_n / (1.0 + 0.6 * cost_n)
    if sacrifice > 2.0:
        sacrifice = 2.0

    # A tiny fresh-order fallback so band-0 orders are still served (keeps volume
    # up, which the fitness multiplies in) but always below any desperate order.
    base = 1.0 + 0.5 * wn

    # Onboard riders: SOFT discount only (rescue tolerates some diversion), never a gate.
    onboard = sum(1 for a in selfd['assigned_order_details'] if a['onboard'])
    onboard_pen = 1.0 / (1.0 + 0.08 * onboard)

    raw = tier + base + 3.0 * sacrifice * (1.0 + 0.5 * band)
    return raw * onboard_pen


def noop_score(driver_obs, phi_ep, phi_step):
    scale = phi_step.mean_solo_time
    if scale <= 0:
        scale = phi_ep.scale
    scale = max(scale, 1e-9)

    selfd = driver_obs['self']
    r = selfd['current_region']

    # Waiting is worthwhile ONLY if we are parked where abandoned demand tends to
    # land (historical od_in) and the region is currently supply-starved -- then we
    # hold the car ready to pounce on the next stale order rather than burning it on
    # a fresh cheap one. The floor is kept BELOW any desperate (band>=1) order value
    # so we never idle past a genuine rescue.
    pull = 0.5
    if r >= 0 and phi_ep.od_orders > 0 and len(phi_ep.od_in) > r:
        pull += 4.0 * phi_ep.od_in[r]
    starved = False
    if r >= 0 and len(phi_step.region_demand) > r and len(phi_step.region_supply) > r:
        if phi_step.region_demand[r] > phi_step.region_supply[r]:
            starved = True
    if starved:
        pull *= 1.6
    else:
        pull *= 0.6

    # Cap the floor well under the desperation tiers (which start at ~1000) so a
    # waiting car always yields to an abandoned order the instant one appears.
    if pull > 3.0:
        pull = 3.0
    return pull
