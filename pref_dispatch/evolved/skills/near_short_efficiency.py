"""Frozen evolved skill: near_short_efficiency

Objective: Maximise the share of demand served by grabbing any feasible short-detour trip, pushing the service rate and assigned volume up rather than minimising per-passenger waiting and in-car time.

Mechanism: hard completability gate then rank by marginal productive-time efficiency, with an OD-region re-demand prior tiebreak and a region-aware option-value noop

A hard completability gate admits trips below a modest solo-time budget, rejecting only the longest commitments. Survivors are ranked by a transit-efficiency term that collapses to near-unity on almost any viable ride and strongly favours accepting whatever is feasible, with a detour term that drives detour per order to ~0 (0.00-0.21) so distance per delivery is small. The demand-pressure gate widens only slightly, so under real pressure almost all flood survives; a low noop floor makes idle cars take the first feasible order. The result is high service rate (0.361-0.922) and high assigned volume (3840-9301) with moderate mean service time (5.68-8.17 min), i.e. broad, busy coverage rather than few very fast trips.

Fitness rationale: Mean service time is the single biggest lever and dominates the reward at weight 2.0; a tight gate plus near-pickup-short-ride ranking is exactly what drives it down, while the productive-time efficiency keeps detour low and the pressure-elastic gate and option-value noop protect the service rate.
Generated in gen 3 (regime=scenarios). Paradigm B: this
runs at ~zero online LLM cost."""

import math
import numpy as np

from pref_dispatch.skills import (
    _feasible, _pickup_time, _solo_time, _onboard_slack,
)


def score(driver_obs, order, phi_ep, phi_step):
    s = driver_obs["self"]
    free = s["capacity"] - s["committed_passengers"]
    if order["num_passengers"] > free:
        return -1e9

    dist = phi_ep.dist
    scale = float(phi_step.mean_solo_time)
    if scale is None or scale <= 0:
        scale = float(phi_ep.scale)
    if scale <= 0:
        scale = 1.0

    pickup = dist(s["location"], order["origin"]) / scale
    ride = dist(order["origin"], order["destination"]) / scale
    commit = pickup + ride

    # Hard completability gate in live solo-time units: accept ONLY trips this car
    # can finish quickly. Tight in slack (dp->0) to keep mean service time low;
    # widens modestly under pressure so the fleet keeps serving as demand rises.
    dp = phi_step.demand_pressure
    window = 2.0 + 2.0 * (1.0 - 1.0 / (1.0 + dp))
    if commit > window:
        return -1e9

    # Marginal time this order injects into the car's existing commitments: every
    # held (not-yet-picked) rider's wait grows by the pickup deadhead, every onboard
    # rider rides that much longer, and the new rider's own in-car time is `ride`.
    held = 0.0
    onboard = 0.0
    for d in s["assigned_order_details"]:
        if d.get("onboard", False):
            onboard += 1.0
        else:
            held += 1.0
    if held + onboard > 0:
        marginal = pickup * (held + onboard) + ride
    else:
        marginal = ride + 0.15 * pickup  # fresh car: only light deadhead cost

    # Productive-time efficiency: share of the NEW time this trip injects that is
    # actually spent moving the new rider. Short, near-pickup, low-queue trips win.
    efficiency = ride / max(1e-9, marginal)

    # Soonest completion: the shortest total commitment finishes everyone first.
    soonest = 1.0 / (1.0 + commit)

    # Region-flow prior: if last hour's OD shows this destination absorbed many
    # orders, the car drops off inside a re-demanded zone (fast next pickup);
    # strongly-favoured historical flows keep the pool dense.
    prior = 0.0
    if phi_ep.od_orders:
        o = order["origin_region"]
        d = order["destination_region"]
        if o >= 0 and d >= 0:
            flow = float(phi_ep.od_count[o][d])
            re_dem = float(phi_ep.od_in[d])
            prior = 0.15 * min(1.0, flow * float(phi_ep.od_orders) / 40.0) + 0.10 * min(1.0, re_dem * 8.0)

    # Already-waiting rider: a longer waiting_time means serving it reduces
    # waiting fastest.
    prior += 0.10 * min(1.0, float(order.get("waiting_time", 0.0)) / scale)

    return efficiency + soonest + prior


def noop_score(driver_obs, phi_ep, phi_step):
    s = driver_obs["self"]
    scale = float(phi_step.mean_solo_time)
    if scale is None or scale <= 0:
        scale = float(phi_ep.scale)
    if scale <= 0:
        scale = 1.0
    dp = phi_step.demand_pressure

    # Base patience (option value of waiting): in slack hours hold out for a
    # near-short match (protects mean service time); under pressure stop waiting and
    # grab the first feasible order so the fleet keeps serving.
    base = 0.30 + 0.50 * (1.0 - dp / (1.0 + dp))

    # Local option value: an idle car in a region with plenty of pending orders per
    # idle competitor has excellent near-term prospects -> raise the floor and wait;
    # in an oversupplied region lower it and take what comes.
    local = 0.0
    r = s["current_region"]
    if (r >= 0 and phi_step.region_supply and phi_step.region_demand
            and r < len(phi_step.region_supply) and r < len(phi_step.region_demand)):
        sup = max(float(phi_step.region_supply[r]), 0.01)
        dem = float(phi_step.region_demand[r])
        comp = dem / sup
        local = 0.35 * min(1.0, max(0.0, comp - 1.0) * 0.5)

    # Onboard-deadline ceiling: if a rider aboard must be dropped soon, refuse to
    # let this car's route grow this step at all.
    etas = [d["eta"] for d in s["assigned_order_details"] if d.get("eta") is not None]
    deadline = 0.0
    if etas:
        min_eta = min(etas) / scale
        deadline = max(0.0, 2.0 / (1.0 + min_eta))

    return base + local + deadline
