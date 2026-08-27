"""Frozen evolved skill: dualfree_ratio_lastseat_topoff

Objective: Maximise seat-occupancy by preferring orders that fill the LAST empty seats of an already-loaded vehicle, driving each car toward its full capacity rather than spreading riders thinly across many cars.

Mechanism: hard dual (tail-fetch + onboard-diversion) gate, then rank survivors by seats-gained-per-marginal-occupied-minute RATIO multiplied by a convex fill-saturation curve with an explicit last-seat close spike

A loaded car keeps its marginal cost tiny: fetch is measured from the nearest committed drop-off (the route tail where the empty seats travel), and any diversion imposed on onboard riders is capped by a hard freeness gate, so only genuinely cheap top-ups survive. Survivors are ranked by seats-gained divided by the marginal occupied vehicle-minutes they add — the exact seat-occupancy ratio the fitness rewards — then multiplied by a cubic fill-fraction curve so the same insert is worth far more the closer it drives the car to full, with an explicit bonus when it closes the last empty seat. An empty car gets only a weak seed (and is penalised for hogging all four seats on a long lone ride), so the dispatcher always prefers to complete a partly-full car over spreading riders thin; a near-full car raises its noop floor and waits, refusing thin far inserts and holding its last seat for the rider that completes it, relaxing this patience only under high demand pressure where throughput matters more.

Fitness rationale: The fixed fitness measures riders-per-vehicle-trip (occupancy) discounted by detour-per-rider, which is exactly maximised when each car is driven full via cheap top-ups rather than scattered thinly, so it faithfully scores this skill's last-seat-closing behaviour.
Generated in gen 5 (regime=scenarios). Paradigm B: this
runs at ~zero online LLM cost."""

import math
import numpy as np

from pref_dispatch.skills import (
    _feasible, _pickup_time, _solo_time, _onboard_slack,
)


def score(driver_obs, order, phi_ep, phi_step):
    me = driver_obs['self']
    cap = me.get('capacity', 1)
    if cap is None or cap < 1:
        cap = 1
    cap = int(cap)
    party = order.get('num_passengers', 1)
    if party is None or party < 1:
        party = 1
    party = int(party)
    load = me.get('committed_passengers', 0)
    if load is None:
        load = 0
    load = int(load)
    free = cap - load
    if party > free:
        return -1e9

    scale = phi_step.mean_solo_time
    if scale is None or scale <= 1e-6:
        scale = phi_ep.scale
    if scale is None or scale <= 1e-6:
        scale = 7.0

    loc = me['location']
    o = order['origin']
    d = order['destination']
    fetch = phi_ep.dist(loc, o)
    ride = phi_ep.dist(o, d)
    if fetch is None:
        fetch = 0.0
    if ride is None:
        ride = 0.0
    fetch_u = fetch / scale
    ride_u = ride / scale

    dp = phi_step.demand_pressure

    details = me.get('assigned_order_details', []) or []
    committed_dests = []
    onboard_drops = []
    for a in details:
        dd = a.get('destination')
        if dd is None:
            continue
        committed_dests.append(dd)
        if a.get('onboard', False):
            onboard_drops.append(dd)

    # --- Parent 1's route-tail anchor: fetch from the FARTHEST committed drop,
    #     the point past which the empty seats keep travelling; take the min of
    #     that and the nearest-tail estimate so cheap detours aren't over-charged.
    tail_fetch_u = fetch_u
    if committed_dests:
        near_tail = min(phi_ep.dist(cd, o) for cd in committed_dests)
        if near_tail is None:
            near_tail = fetch
        tail_fetch_u = max(0.0, near_tail) / scale

    # diversion imposed on onboard riders (must stay bounded).
    onboard_detour_u = 0.0
    if onboard_drops:
        nd = min(phi_ep.dist(dp2, o) for dp2 in onboard_drops)
        if nd is None:
            nd = 0.0
        onboard_detour_u = max(0.0, nd) / scale

    # --- hard pickup-reach gate: tighter for a loaded car so it PROTECTS its
    #     fullness (Parent 1), relaxed under demand pressure (Parent 2). ---
    reach_cap = 2.2 if load == 0 else 1.4
    if dp is not None and dp > 1.0:
        reach_cap += 0.6 * min(2.0, dp - 1.0)
    gate_fetch = fetch_u if load == 0 else tail_fetch_u
    if gate_fetch > reach_cap:
        return -1e9

    # --- loaded freeness gate: onboard diversion is a small slice only. ---
    if load > 0:
        free_cap = 1.1
        if dp is not None and dp > 1.0:
            free_cap += 0.5 * min(2.0, dp - 1.0)
        if onboard_detour_u > free_cap:
            return -1e9

    frac_now = load / float(cap)
    post_fill = (load + party) / float(cap)

    # --- CORE RATIO: seats gained per marginal occupied vehicle-minute ---
    if load == 0:
        added_minutes = fetch_u + ride_u
    else:
        ride_share = 0.5 * (1.0 - frac_now)
        added_minutes = tail_fetch_u + ride_share * ride_u
    eps = 0.15
    ratio = party / (eps + added_minutes)

    # --- STEEP convex fill saturation: last seat worth far more than first. ---
    if load == 0:
        fill_reward = 0.15 * post_fill
    else:
        fill_reward = pow(post_fill, 3.0)
        # explicit close-the-car spike: filling the last empty seat is the point.
        if load + party == cap:
            fill_reward += 0.7
        elif post_fill >= 0.75:
            fill_reward += 0.15
        # extra credit for consuming a LARGE share of the remaining empty seats
        # (Parent 1's gap-closed term) so partial top-ups toward full beat thin ones.
        gap_closed = party / float(free)
        fill_reward *= (1.0 + 0.4 * gap_closed)

    # --- smooth detour attenuation on the added marginal minutes. ---
    attenuation = 1.0 / (1.0 + 0.8 * added_minutes * added_minutes)

    value = 1000.0 * ratio * fill_reward * attenuation

    # empty-car anti-hogging: a lone long ride occupying all seats is anti-pooling.
    if load == 0:
        if ride_u > 2.5:
            value -= 40.0 * (ride_u - 2.5)
        value -= 1.5 * ride_u
    else:
        # mild preference for shorter added rides so a full car stays fast.
        value -= 1.0 * ride_u

    # tiny throughput nudge so serving beats not-serving at the margin.
    value += 1.0
    return float(value)


def noop_score(driver_obs, phi_ep, phi_step):
    me = driver_obs['self']
    cap = me.get('capacity', 1)
    if cap is None or cap < 1:
        cap = 1
    cap = int(cap)
    load = me.get('committed_passengers', 0)
    if load is None:
        load = 0
    load = int(load)
    if load == 0:
        return 0.0
    if load >= cap:
        return 1e9
    fill = load / float(cap)
    remaining = cap - load
    # convex floor climbs with fill so a near-full car refuses thin far inserts
    # and holds out for the rider that closes it; spikes when one seat remains.
    floor = 1000.0 * pow(fill, 3.0) * 0.33
    if remaining == 1:
        floor += 65.0
    # relax under high demand pressure: throughput matters more than perfect fill.
    dp = phi_step.demand_pressure
    if dp is not None and dp > 1.0:
        floor *= 1.0 / (1.0 + 0.5 * min(2.0, dp - 1.0))
    return float(floor)
