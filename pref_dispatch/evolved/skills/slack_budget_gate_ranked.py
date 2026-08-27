"""Frozen evolved skill: slack_budget_gate_ranked

Objective: Minimise mean end-to-end passenger service time by only pooling a new rider when it barely delays the passengers already onboard, otherwise taking fast solo trips.

Mechanism: hard per-passenger delay-budget gate (empty vs loaded car) then rank survivors by absolute added service time, smallest first

For an empty car the order is a pure solo trip, so it is always feasible and is ranked by how fast it can be served (short pickup + short ride win) — a solo trip incurs zero pooling delay. For a loaded car the score estimates the marginal detour the new pickup-then-drop insertion adds to the existing route and hard-rejects the order if that detour would push any onboard passenger's extra wait past a tight budget (a small fraction of one solo trip), scaled a little looser only when demand pressure is high so throughput does not collapse. Survivors are ranked by the NEGATIVE total added service minutes, so the cheapest-to-insert rider always wins. It prefers to wait (positive noop floor) when a car already carries riders whose slack is nearly spent, refusing any marginally-delaying pool and holding capacity for a genuinely free insert.

Fitness rationale: Mean service time is the direct objective; the detour term penalises pooling that lengthens onboard riders' journeys, and the completed term guards against a gate so strict it serves nothing.
Generated in gen 5 (regime=scenarios). Paradigm B: this
runs at ~zero online LLM cost."""

import math
import numpy as np

from pref_dispatch.skills import (
    _feasible, _pickup_time, _solo_time, _onboard_slack,
)


def score(driver_obs, order, phi_ep, phi_step):
    me = driver_obs['self']
    cap = me['capacity']
    if cap <= 0:
        return -1e9
    # live scale unit
    scale = phi_step.mean_solo_time
    if scale is None or scale <= 1e-6:
        scale = phi_ep.scale
    if scale <= 1e-6:
        scale = 1.0

    o_org = order['origin']
    o_dst = order['destination']
    party = order.get('num_passengers', 1)
    loc = me['location']
    dist = phi_ep.dist

    details = me.get('assigned_order_details', [])

    # count committed load
    onboard_load = 0
    onboard = []
    for d in details:
        if d.get('onboard', False):
            onboard_load += d.get('num_passengers', 1)
            onboard.append(d)
    committed = me.get('committed_passengers', onboard_load)

    # capacity feasibility
    if committed + party > cap:
        return -1e9

    # pickup deadhead from current location
    fetch = dist(loc, o_org)
    ride = dist(o_org, o_dst)

    # EMPTY car: pure solo trip, zero pooling delay -> always feasible
    if onboard_load == 0 and len(details) == 0:
        # rank by how fast we can serve: short pickup + short ride win
        total = fetch + ride
        return 2.0 * scale - total / scale

    # LOADED (or committed) car: charge marginal detour on onboard riders.
    # Estimate added service minutes: the new pickup+drop insertion forces the
    # car to divert. Approximate the per-onboard extra wait as the detour of
    # visiting the new origin then destination relative to going straight.
    # marginal added travel = fetch + ride + tail - straight
    # Use the last known onboard destination as the straight continuation.
    ref_dst = None
    max_eta = -1.0
    for d in onboard:
        e = d.get('eta', 0.0)
        if e is not None and e > max_eta:
            max_eta = e
            ref_dst = d.get('destination', None)
    if ref_dst is None:
        # fall back to any assigned detail
        for d in details:
            if d.get('destination') is not None:
                ref_dst = d['destination']
                break
    if ref_dst is None:
        ref_dst = o_dst

    straight = dist(loc, ref_dst)
    detour_path = fetch + ride + dist(o_dst, ref_dst)
    added = detour_path - straight
    if added < 0.0:
        added = 0.0

    # per-onboard extra wait budget: tight fraction of a solo trip.
    # loosen a little when demand pressure is high (throughput protection).
    dp = phi_step.demand_pressure
    budget_frac = 0.20
    if dp is not None and dp > 1.0:
        budget_frac = 0.20 + 0.15 * min(dp - 1.0, 2.0)
    budget = budget_frac * scale

    # each onboard passenger roughly eats `added` extra minutes; hard gate
    if added > budget:
        return -1e9

    # survivors ranked by smallest total added minutes (fetch also matters)
    total_added = added + 0.5 * fetch
    return 1.0 * scale - total_added / scale


def noop_score(driver_obs, phi_ep, phi_step):
    me = driver_obs['self']
    details = me.get('assigned_order_details', [])
    onboard = [d for d in details if d.get('onboard', False)]
    if not onboard:
        # idle / free car: no reason to hold out, floor at zero
        return 0.0
    scale = phi_step.mean_solo_time
    if scale is None or scale <= 1e-6:
        scale = phi_ep.scale
    if scale <= 1e-6:
        scale = 1.0
    # If onboard riders are close to their drop (small eta), their slack is
    # nearly spent: hold out, refuse any marginally-delaying pool.
    min_eta = min((d.get('eta', 0.0) or 0.0) for d in onboard)
    tightness = 1.0 - min(min_eta / scale, 1.0)
    # positive floor rises as onboard slack shrinks
    return 0.30 * scale * tightness / scale
