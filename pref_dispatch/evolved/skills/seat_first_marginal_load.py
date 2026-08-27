"""Frozen evolved skill: seat_first_marginal_load

Objective: Maximise the number of distinct riders served this hour by grabbing every feasible order when a seat is free, sacrificing per-fare revenue and detour efficiency to stop riders going unmatched.

Mechanism: marginal/counterfactual cost: a saturating base value for 'one more rider served' minus the marginal detour burden this insertion places on the riders already committed to the vehicle, ranked against an inverted-patience noop floor

Any order that physically fits the free seats is worth a fixed, saturating value — one served rider is one served rider, so there is no fare or length preference. What separates candidates is only the burden the insertion puts on riders already committed: an empty car has nobody to delay, so its burden is zero and it grabs the first feasible order in the step; a vehicle already carrying passengers pays a burden equal to the pickup (and ride detour) weighted by its load, so in a slack market a near-full car only inserts very close pickups and otherwise prefers to wait. Demand pressure is the relaxer in both directions: when riders are piling up unmatched, the burden and the waiting floor both collapse, so even a loaded car takes every feasible seat-filler rather than hold out. The noop floor is deliberately tiny when a car is empty — waiting costs a served rider — and it grows with load (protecting banked service during a slack hour) but shrinks under pressure, so the mechanism free-runs between 'grab everything' and 'protect what's on board' purely off the scene's own load, time and pressure scales.

Fitness rationale: Service_rate is the direct measure of the objective — the fraction of riders served rather than matched/revenue — and the assigned-volume term in mean_solo-time-scaled units rewards the raw number of riders moved, so a skill that grabs every feasible order scores high on both.
Generated in gen 3 (regime=scenarios). Paradigm B: this
runs at ~zero online LLM cost."""

import math
import numpy as np

from pref_dispatch.skills import (
    _feasible, _pickup_time, _solo_time, _onboard_slack,
)


def score(driver_obs, order, phi_ep, phi_step):
    self = driver_obs['self']
    cap = self['capacity']
    party = order['num_passengers']
    committed = self['committed_passengers']
    free = cap - committed
    if free <= 0 or party > free:
        return -1e9

    scale = phi_step.mean_solo_time
    if scale is None or scale <= 0:
        scale = phi_ep.scale
    if scale is None or scale <= 0:
        scale = 1.0

    pickup = phi_ep.dist(self['location'], order['origin']) / scale
    ride = phi_ep.dist(order['origin'], order['destination']) / scale

    # Saturation: any feasible order is a fully served rider --- value plateaus.
    base = 1.0

    # Marginal / counterfactual: burden on riders ALREADY committed.
    load = float(committed) / float(cap)
    onboard = 0
    for od in self['assigned_order_details']:
        if od.get('onboard', False):
            onboard += 1

    # Demand pressure relaxes the burden: unmatched riders make even a loaded
    # car take a feasible seat-filler instead of holding out.
    pressure = phi_step.demand_pressure
    relax = 1.0 / (1.0 + pressure)

    burden = 2.5 * load * pickup * relax
    if onboard > 0:
        burden += 0.6 * load * ride * relax

    return base - burden


def noop_score(driver_obs, phi_ep, phi_step):
    self = driver_obs['self']
    cap = self['capacity']
    committed = self['committed_passengers']
    free = cap - committed
    scale = phi_step.mean_solo_time
    if scale is None or scale <= 0:
        scale = phi_ep.scale
    if scale is None or scale <= 0:
        scale = 1.0

    if free <= 0:
        return 1.5

    load = float(committed) / float(cap)
    pressure = phi_step.demand_pressure
    # Empty cars have nothing to protect -> grab anything (floor ~0).
    # Loaded cars' floor rises with load -> wait unless a cheap insert appears.
    floor = 0.02 + 0.7 * load
    # Unmatched riders destroy the luxury of patience -> floor collapses.
    floor = floor / (1.0 + 1.5 * pressure)
    return max(0.0, floor)
