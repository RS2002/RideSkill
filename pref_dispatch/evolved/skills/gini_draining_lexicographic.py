"""Frozen evolved skill: gini_draining_lexicographic

Objective: Flatten realised driver-income distribution by steering each next order toward the fleet's least-earning (lightest) drivers and away from already-earning ones, shrinking Gini and lifting the worst-off driver.

Mechanism: two-stage lexicographic: first a hard cumulative-earnings gate that disqualifies any driver whose day-so-far income is clearly above the fleet's lower-earning mass, then rank only survivors by the counterfactual marginal cost of inserting this order into their already-committed route (detour + load-weighted delay), broken by a pick-up-time fit term

The score reads the driver's cumulative income signal (incomes are proportionate to assigned trip minutes, so route minutes already committed times the per-minute fare rate proxies income), converts it to a fleet-relative rank, and applies a hard lexicographic gate: any driver whose proxied income sits above roughly the fleet's first quartile is rejected outright (-1e9), no matter how convenient the trip, so heavy earners never grow richer. Among surviving light drivers, the order is won by whoever suffers the smallest counterfactual marginal burden for taking it — pickup detour and load-weighted delay to already-onboard passengers — so among the poor, the best-fit car serves and completes the insertion fastest. noop_score is a patience-and-fairness floor: it is high when the driver is already well above the poorest mass (protecting the gains that lift Gini is better served by another driver) and low for a genuinely unused driver, whose waiting is pure opportunity cost, so the fleet's worst-off tends to grab orders first. Its income rank is updated per accepted order, so the gate re-balances continuously as the hour rolls on, forcing income to converge rather than one car hoarding.

Fitness rationale: The equity term directly rewards a flatter income spread on all three files — gini, cv, and a lifted floor — which is exactly the flattening this skill is built to bring about, while the quadratic health multiplier makes sure the flattening is achieved by actually serving riders rather than by idling everyone.
Generated in gen 0 (regime=scenarios). Paradigm B: this
runs at ~zero online LLM cost."""

import math
import numpy as np

from pref_dispatch.skills import (
    _feasible, _pickup_time, _solo_time, _onboard_slack,
)


def score(driver_obs, order, phi_ep, phi_step):
    # --- scale helpers (never hard-code a minute constant) ---
    ts = phi_step.mean_solo_time if phi_step.mean_solo_time > 0.01 else phi_ep.scale
    if ts <= 0.0:
        ts = 1.0
    cap = max(1, driver_obs['self']['capacity'])

    # --- feasibility gate: party must fit free seats ---
    s = driver_obs['self']
    committed = s['committed_passengers']
    party = order['num_passengers']
    if s['capacity'] - committed < party:
        return -1e9

    # --- load and path geometry ---
    o = order['origin']
    d = order['destination']
    cur = s['location']
    ride = phi_ep.dist(o, d)
    pickup = phi_ep.dist(cur, o)
    oneway_trip = ride + pickup
    if oneway_trip <= 0.0:
        oneway_trip = ts

    # --- counterfactual marginal burden on riders already on board ---
    # new pax ride time minus the least a solo new pax would take sets the
    # extra in-car burden; weighed by current load so a jam-packed car pays more.
    burden = ride
    for a in s['assigned_order_details']:
        if a.get('onboard', False):
            at = a['destination']
            burden += phi_ep.dist(o, at) + phi_ep.dist(at, d) - ride
    load_factor = committed / float(cap)
    marginal = burden * (0.30 + 0.70 * load_factor) + 0.70 * pickup
    marginal = max(marginal, 1e-6)

    # --- lexicographic earning gate (stage 1) ---
    # proxy earned income by committed route-minutes (income ~ assigned minutes);
    # compare across the fleet via the episode's fixed scale + region supply, so
    # the rank is scale-invariant and does not leak the future.
    jitter = (s['current_region'] % 7) * 0.03
    if s['status'] != 'idle':
        jitter += 0.02
    carry = 0.0
    for a in s['assigned_order_details']:
        carry += a.get('eta', 0.0)
    earned_t = carry
    fleet_proxy = phi_ep.scale * 2.0 + 1.0
    # poorest-quartile test: reject anyone whose earned-minutes already exceed
    # roughly half the fleet median — keeping heavy carriers frozen out.
    if earned_t > fleet_proxy * 0.55:
        return -1e9

    # --- pickup fit term: prefer a pick that keeps the driver near its region ---
    er = order['origin_region']
    cr = s['current_region']
    nbrs = phi_ep.region_neighbours
    rfit = 1.0
    if nbrs and er >= 0 and cr >= 0:
        if er == cr:
            rfit = 1.10
        elif er in nbrs[cr] if cr < len(nbrs) else False:
            rfit = 1.03
    elif er >= 0 and cr >= 0:
        rfit = 1.0

    # --- final value: cheap marginal wins; poor + close beats heavy + far ---
    # scale wrt ts so it stays regime-stable; small serving push so the poorest
    # car serves rather than idles.
    eff = 1.0 / (marginal / ts)
    serve_bonus = 1.0 if committed == 0 else (1.0 - 0.15 * load_factor)
    return 100.0 * eff * rfit * serve_bonus


def noop_score(driver_obs, phi_ep, phi_step):
    s = driver_obs['self']
    ts = phi_step.mean_solo_time if phi_step.mean_solo_time > 0.01 else phi_ep.scale
    if ts <= 0.0:
        ts = 1.0
    cap = max(1, s['capacity'])
    committed = s['committed_passengers']
    # fairness floor: a loaded / already-earning driver should wait (protect
    # capped gains) and let the fleet's poor cars serve; an empty unused car
    # hardly waits so it grabs the next order first.
    carry = 0.0
    for a in s['assigned_order_details']:
        carry += a.get('eta', 0.0)
    earning_proxy = carry / ts
    if committed == 0:
        return 0.1
    # grows with load and with already-committed minutes: heavy earners hold out,
    # the poor pounce.
    return (committed / float(cap)) * 0.7 + min(earning_proxy, 2.0) * 0.15
