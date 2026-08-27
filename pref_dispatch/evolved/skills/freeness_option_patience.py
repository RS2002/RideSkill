"""Frozen evolved skill: freeness_option_patience

Objective: Minimise pooling detour cost per additional rider by only accepting a new order when the driver's route already passes very close to BOTH its pickup and its drop-off, so the extra rider is essentially free to serve.

Mechanism: patience / option value: a high noop floor that makes waiting a real choice, so the driver holds out for a genuinely dual-free insert rather than accepting any feasible order; the score is the ratio of party size to marginal detour cost, gated by a strict dual-proximity hard reject

For a loaded car the skill computes the minimum true insertion detour across all consecutive anchor pairs, hard-rejects any order where either endpoint is far from the route or the combined detour exceeds a pressure-scaled fraction of one solo trip, then ranks survivors by the ratio party / (epsilon + best_detour) — a pure efficiency ratio that gives near-infinite value to genuinely free inserts and collapses quickly as detour grows. The noop floor is set high enough that a loaded car will not accept an order whose detour ratio merely beats zero; it must beat a floor that rises with fill and falls with demand pressure, so thin-fleet high-pressure scenes still serve riders while well-supplied scenes hold out for free pools. An empty car always accepts, ranked by inverse fetch distance, since there is no pooling detour to protect.

Fitness rationale: The ratio directly measures the skill's core trade-off: completed riders in the numerator rewards throughput while the detour penalty in the denominator suppresses pooling cost, so variants that serve more riders with less added route length score higher.
Generated in gen 5 (regime=scenarios). Paradigm B: this
runs at ~zero online LLM cost."""

import math
import numpy as np

from pref_dispatch.skills import (
    _feasible, _pickup_time, _solo_time, _onboard_slack,
)


def score(driver_obs, order, phi_ep, phi_step):
    self = driver_obs['self']
    cap = self['capacity']
    if cap is None or cap < 1:
        cap = 1
    party = order.get('num_passengers', 1)
    committed = self.get('committed_passengers', 0)
    if committed + party > cap:
        return -1e9

    scale = phi_step.mean_solo_time
    if scale is None or scale <= 1e-6:
        scale = phi_ep.scale
    if scale is None or scale <= 1e-6:
        scale = 1.0

    d = phi_ep.dist
    loc = self['location']
    o_pick = order['origin']
    o_drop = order['destination']
    solo = d(o_pick, o_drop)

    # Empty car: seed a solo trip ranked by inverse fetch
    if committed <= 0:
        fetch = d(loc, o_pick)
        return party / (1.0 + fetch / scale)

    # Build anchor list: location, pending pickups, then drops
    details = self.get('assigned_order_details', []) or []
    anchors = [loc]
    for a in details:
        if not a.get('onboard', False):
            ap = a.get('origin')
            if ap is not None:
                anchors.append(ap)
    for a in details:
        ad = a.get('destination')
        if ad is not None:
            anchors.append(ad)

    # Pressure-scaled gate: tighter when pressure is low (free inserts plentiful)
    pressure = phi_step.demand_pressure
    if pressure is None or pressure < 0:
        pressure = 0.0
    # gate relaxes from 0.20 -> 0.35 as pressure rises from 0 -> 2
    gate_frac = 0.20 + 0.075 * min(pressure, 2.0)
    end_max = gate_frac * scale

    pick_touch = min(d(anc, o_pick) for anc in anchors)
    drop_touch = min(d(anc, o_drop) for anc in anchors)
    if pick_touch > end_max or drop_touch > end_max:
        return -1e9
    if pick_touch + drop_touch > 1.5 * end_max:
        return -1e9

    # True marginal insertion detour across all consecutive anchor pairs
    best_detour = pick_touch + solo + drop_touch
    n = len(anchors)
    for i in range(n):
        start = anchors[i]
        if i + 1 < n:
            end = anchors[i + 1]
            det = d(start, o_pick) + solo + d(o_drop, end) - d(start, end)
        else:
            det = d(start, o_pick) + solo
        if det < best_detour:
            best_detour = det
    if best_detour < 0.0:
        best_detour = 0.0

    # Efficiency ratio: party per unit of marginal detour
    eps = 1e-3 * scale
    return party / (eps + best_detour)


def noop_score(driver_obs, phi_ep, phi_step):
    self = driver_obs['self']
    committed = self.get('committed_passengers', 0)
    if committed <= 0:
        return 0.0
    cap = self['capacity']
    if cap is None or cap < 1:
        cap = 1
    fill = committed / cap
    pressure = phi_step.demand_pressure
    if pressure is None or pressure < 0:
        pressure = 0.0
    scale = phi_step.mean_solo_time
    if scale is None or scale <= 1e-6:
        scale = phi_ep.scale
    if scale is None or scale <= 1e-6:
        scale = 1.0
    # Floor rises with fill (heavier car waits harder) and falls with pressure
    # (high pressure = scarce free inserts, lower patience to protect throughput)
    patience = (0.4 + 0.4 * fill) / (1.0 + 0.5 * min(pressure, 4.0))
    # Express floor in same units as score: party / detour ~ 1 / (frac * scale)
    return patience / (0.05 * scale)
