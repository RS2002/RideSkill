"""Frozen evolved skill: gowin_ratio_gate

Objective: Maximise realised revenue per served rider by aggressively preferring long-fare, high-value trips even at the cost of extra pickup distance.

Mechanism: hard window gate that breathes with demand pressure, then fare-per-pickup-minute ratio ranking among survivors

After a hard feasibility gate (party fits free capacity), the score rejects any order whose total commitment (pickup + ride) blows a dispatch window expressed in live mean-solo-time units that widens with demand pressure -- so in a slack hour it only commits to trips it can finish comfortably, while in a congested hour it relaxes to keep serving. Survivors are ranked by marginal fare per marginal pickup minute: the fare model is monotone in ride length and party size and is divided by the pickup cost, so a long high-value fare with a far pickup is aggressively preferred, while a short cheap fare is deprioritised. A tiny pressure-driven nudge stops zero-pickup blow-ups, a small passenger bonus tilts toward pooling more riders, an OD-flow bonus breaks ties toward historically flowing routes, and a pickup drag keeps pathological deadheads in check. The noop/waiting floor is a route-aware estimate of the typical future fare-per-pickup this driver can expect from its own region (boosted where that region was a hot origin last hour), scaled by a patience factor that is high when supply is ample (low pressure) so the driver holds out for a better match and low when supply is scarce so it grabs what comes; tight onboard deadlines raise the floor so it never risks a route that could miss riders it already has.

Fitness rationale: The fitness pays realised revenue directly, scaled up by how many riders actually got served (service_rate, weighted 0.7) so revenue-hungry skills cannot starve the fleet, and divided by income variance so it also rewards not concentrating all money on a few drivers. Chasing ratio-max long fares raises revenue; the pressure-adaptive gate protects the service-rate and income_cv terms by never deadheading catastrophically.
Generated in gen 1 (regime=scenarios). Paradigm B: this
runs at ~zero online LLM cost."""

import math
import numpy as np

from pref_dispatch.skills import (
    _feasible, _pickup_time, _solo_time, _onboard_slack,
)


def score(driver_obs, order, phi_ep, phi_step):
    try:
        selfd = driver_obs['self']
        cap = selfd['capacity']
        party = order['num_passengers']
        if party < 1:
            return -1e9
        if party > (cap - selfd.get('committed_passengers', 0)):
            return -1e9
        # live scale, falling back to a static map scale only when empty.
        scale = phi_step.mean_solo_time
        if scale is None or scale <= 1e-9:
            scale = phi_ep.scale
        if scale is None or scale <= 1e-9:
            scale = 1.0
        dist = phi_ep.dist
        pickup_u = max(dist(selfd['location'], order['origin']), 1e-6) / scale
        ride_u = max(dist(order['origin'], order['destination']), 1e-6) / scale
        pressure = phi_step.demand_pressure
        if pressure is None:
            pressure = 0.0
        # HARD WINDOW GATE (Parent2's guard, made to breathe with pressure like
        # Parent1's gate): bound in units of live mean solo time -- tight (6x) when
        # demand slacks so we stay picky, looser when congested to avoid starvation.
        if pickup_u + ride_u > 6.0 + 4.0 * min(1.0, pressure):
            return -1e9
        # RATIO RANK (Parent2's signature): marginal fare per marginal pickup time.
        # Monotone in ride length and party size -> aggressively favours long,
        # high-value fares even at extra pickup cost.
        fare_per_rider = 2.5 + 2.2 * ride_u
        fare_total = fare_per_rider * party
        nudge = 0.15 / (1.0 + 0.5 * pressure)   # guards against ~0-pickup blow-ups
        pickup_e = pickup_u + nudge
        ratio = fare_total / pickup_e
        # mild passenger bonus, small OD-flow tie-break (Parent1), pickup drag.
        bonus = 0.0
        try:
            if phi_ep.od_orders:
                orr = order.get('origin_region', -1)
                dr = order.get('destination_region', -1)
                if orr >= 0 and dr >= 0:
                    bonus = 2.0 * phi_ep.od_count[orr][dr]
        except Exception:
            bonus = 0.0
        onroute = 0.0
        try:
            for a in selfd.get('assigned_order_details', []):
                if a.get('onboard', False):
                    onroute += 0.05 * a['num_passengers']
        except Exception:
            onroute = 0.0
        return ratio * (1.0 + 0.5 * party) + bonus - onroute - 0.2 * pickup_u
    except Exception:
        return -1e9


def noop_score(driver_obs, phi_ep, phi_step):
    try:
        selfd = driver_obs['self']
        scale = phi_step.mean_solo_time
        if scale is None or scale <= 1e-9:
            scale = phi_ep.scale
        if scale is None or scale <= 1e-9:
            scale = 1.0
        pressure = phi_step.demand_pressure
        if pressure is None:
            pressure = 0.0
        # Typical future match from here: ~1 solotime ride, ~half a solotime pickup,
        # in the same fare-per-pickup units the score uses so they compare directly.
        nudge = 0.15 / (1.0 + 0.5 * pressure)
        pickup_e = 0.5 + nudge
        future_ratio = (2.5 + 2.2 * 1.0) / pickup_e
        # Patience / option value: ample supply (low pressure) -> can afford to wait
        # (raise the floor); scarce supply -> grab what comes (lower it). Route-aware
        # boost where this region was a hot origin last hour (better future odds).
        patience = 0.35 + 0.35 * (1.0 - min(1.0, pressure))
        try:
            if phi_ep.od_orders:
                reg = selfd.get('current_region', -1)
                if reg >= 0 and reg < len(phi_ep.od_out):
                    patience += 0.3 * phi_ep.od_out[reg]
        except Exception:
            pass
        floor = future_ratio * patience
        # Onboard-slack urgency: tight deadlines for already-grabbed riders raise the
        # floor so the driver won't risk a route that could miss them.
        etas = [a['eta'] for a in selfd.get('assigned_order_details', []) if a.get('eta') is not None]
        if etas:
            slack_u = min(etas) / scale
            floor += 3.0 / (1.0 + max(0.0, slack_u))
        return floor
    except Exception:
        return 0.0
