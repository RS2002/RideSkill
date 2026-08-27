"""Frozen Feature-3 reposition scorer: dualmech_od_reach_hybrid_scorer

Objective: Score each candidate region by objective-priced, supply-netted near-future demand (live eff_demand plus seat/OD-weighted pending pool) minus a fairness-reach-scaled, empty-averse-stiffened cruise penalty, switching emphasis by what `w` pays for.

It buckets live pending orders and eff_demand by region, nets against neighbourhood free supply, and adds an OD-inflow structural prior weighted by demand-pressure. Parent A's mechanism (stiff empty-averse penalty, boosted-rival discount, and a park margin that rises with cruise price) governs costly-cruise / strong-fairness cells; parent B's mechanism (OD-inflow prior scaled by od_orders and a demand-multiplier from the fairness budget) governs raw/completion cells. Fairness reach extends demand-chasing for boosted (poor) cars and damps rich cars parked on hotspots at high strength; it returns {} to keep a car put whenever the best move fails to beat staying by a cruise-priced margin.

Reposition understanding (LLM CoT): A region is worth cruising an idle empty car toward when it has genuine near-future pickup demand (live under-served eff_demand or a habitual origin per the OD prior) that this car can reach before the demand evaporates, and where free supply is not already plentiful. A service-maximising scorer prices that demand in the units the current objective `w` actually pays (finished trips, seats, or raw throughput), subtracts an empty-cruise penalty scaled by how costly `w` makes empty moves and by the driver's fairness reach, and refuses to move when the best target barely beats staying put.

Evolved under GROUP-RELATIVE fitness: on each (scene, objective,
fairness-strength) cell, (my reward - the cell's mean) / the cell's
spread, where the cell also holds the demand-gravity heuristic and
repositioning switched OFF, gen 7. Authors ONLY per-region
base scores: coordinated spreading, stay rules, and the relocate action
stay in pref_dispatch.reposition."""

import math
import numpy as np


def reposition_scores(driver_obs, phi_ep, phi_step, kappa, w):
    self = driver_obs.get('self', {})
    cur = self.get('current_region', -1)
    if cur is None or cur < 0:
        return {}
    rp = driver_obs.get('relocation_points', ())
    n = len(rp)
    if n == 0:
        return {}
    loc = self.get('location')
    scale = None
    try:
        scale = float(phi_step.mean_solo_time)
    except:
        scale = None
    if scale is None or scale <= 1e-6:
        try:
            scale = float(phi_ep.scale)
        except:
            scale = None
    if scale is None or scale <= 1e-6:
        scale = 7.0
    scale = float(scale)
    trip_val = 1.0
    empty_cost = 0.30
    seat_val = 0.0
    assign_val = 0.0
    if w is not None:
        try:
            base = float(w({'assigned_orders': [], 'assigned_party_sizes': {},
                     'assigned_solo_times': {}, 'assigned_service_times': {},
                     'completed_orders': [], 'picked_up_orders': [],
                     'distance_moved': 0.0, 'time_moved': 0.0,
                     'is_empty_move': False, 'is_idle_wait': False,
                     'extra_detour_time': 0.0}))
            done = float(w({'assigned_orders': [1], 'assigned_party_sizes': {1: 1},
                     'assigned_solo_times': {1: scale}, 'assigned_service_times': {1: scale},
                     'completed_orders': [1], 'picked_up_orders': [1],
                     'distance_moved': 0.0, 'time_moved': 0.0,
                     'is_empty_move': False, 'is_idle_wait': False,
                     'extra_detour_time': 0.0}))
            trip_val = abs(done - base)
            if not trip_val > 1e-9:
                trip_val = 1.0
            emv = float(w({'assigned_orders': [], 'assigned_party_sizes': {},
                     'assigned_solo_times': {}, 'assigned_service_times': {},
                     'completed_orders': [], 'picked_up_orders': [],
                     'distance_moved': scale, 'time_moved': scale,
                     'is_empty_move': True, 'is_idle_wait': False,
                     'extra_detour_time': 0.0}))
            ec = base - emv
            if ec > 0:
                empty_cost = ec
            seat = float(w({'assigned_orders': [1], 'assigned_party_sizes': {1: 3},
                     'assigned_solo_times': {1: scale}, 'assigned_service_times': {1: scale},
                     'completed_orders': [1], 'picked_up_orders': [1],
                     'distance_moved': 0.0, 'time_moved': 0.0,
                     'is_empty_move': False, 'is_idle_wait': False,
                     'extra_detour_time': 0.0}))
            seat_val = abs(seat - base) - trip_val
            if seat_val < 0:
                seat_val = 0.0
            asg = float(w({'assigned_orders': [1], 'assigned_party_sizes': {1: 1},
                     'assigned_solo_times': {1: scale}, 'assigned_service_times': {1: scale},
                     'completed_orders': [], 'picked_up_orders': [],
                     'distance_moved': 0.0, 'time_moved': 0.0,
                     'is_empty_move': False, 'is_idle_wait': False,
                     'extra_detour_time': 0.0}))
            assign_val = abs(asg - base)
        except:
            trip_val = 1.0
            empty_cost = 0.30
            seat_val = 0.0
            assign_val = 0.0
    cruise_w = empty_cost / trip_val
    if cruise_w < 0.05:
        cruise_w = 0.05
    if cruise_w > 4.0:
        cruise_w = 4.0
    empty_averse = cruise_w > 0.5
    seat_frac = seat_val / trip_val
    if seat_frac < 0:
        seat_frac = 0.0
    if seat_frac > 2.0:
        seat_frac = 2.0
    thru_frac = assign_val / trip_val
    if thru_frac < 0:
        thru_frac = 0.0
    if thru_frac > 2.0:
        thru_frac = 2.0
    fb = driver_obs.get('fairness_budget', 1.0)
    try:
        fb = float(fb)
    except:
        fb = 1.0
    if fb <= 0:
        fb = 1.0
    fs = 0.0
    try:
        fs = float(phi_ep.fairness_strength)
    except:
        fs = 0.0
    reach = 1.0
    if fs > 0.0:
        reach = 1.0 + 0.6 * (fb - 1.0) * min(fs, 2.0)
        if reach < 0.4:
            reach = 0.4
        if reach > 2.5:
            reach = 2.5
    dem_mult = 1.0
    if fs > 0.0:
        dem_mult = 1.0 + 0.6 * fs * (fb - 1.0)
        if dem_mult < 0.3:
            dem_mult = 0.3
        if dem_mult > 2.0:
            dem_mult = 2.0
    dbud = driver_obs.get('driver_budgets', {})
    alldr = driver_obs.get('all_drivers', {})
    od_in = None
    od_out = None
    od_orders = 0.0
    try:
        if phi_ep.od_orders and float(phi_ep.od_orders) > 0:
            od_in = phi_ep.od_in
            od_out = phi_ep.od_out
            od_orders = float(phi_ep.od_orders)
    except:
        od_in = None
        od_out = None
        od_orders = 0.0
    dp = 0.0
    try:
        dp = float(phi_step.demand_pressure)
    except:
        dp = 0.0
    if dp < 0:
        dp = 0.0
    dp_ramp = dp / (dp + 2.0)
    if dp_ramp > 1.0:
        dp_ramp = 1.0
    pend = driver_obs.get('pending_orders', [])
    reg_pax = {}
    reg_cnt = {}
    reg_multi = {}
    for o in pend:
        try:
            orr = o.get('origin_region', -1)
        except:
            continue
        if orr is None or orr < 0:
            continue
        try:
            npax = float(o.get('num_passengers', 1))
        except:
            npax = 1.0
        if npax < 1:
            npax = 1.0
        reg_pax[orr] = reg_pax.get(orr, 0.0) + npax
        reg_cnt[orr] = reg_cnt.get(orr, 0.0) + 1.0
        if npax >= 2.0:
            reg_multi[orr] = reg_multi.get(orr, 0.0) + npax
    nb = driver_obs.get('region_neighbours', ())
    cands = set()
    cands.add(cur)
    try:
        if cur < len(nb):
            for x in nb[cur]:
                if 0 <= x < n:
                    cands.add(x)
    except:
        pass
    try:
        ed = kappa.eff_demand
        order = sorted(range(n), key=lambda r: -float(ed[r]))
        for r in order[:6]:
            cands.add(r)
    except:
        pass
    def cap_supply(r):
        try:
            return float(kappa.supply[r])
        except:
            return 0.0
    def cap_eff(r):
        try:
            return float(kappa.eff_demand[r])
        except:
            return 0.0
    def nb_supply(r):
        s = cap_supply(r)
        try:
            if r < len(nb):
                for x in nb[r]:
                    if 0 <= x < n:
                        s += 0.5 * cap_supply(x)
        except:
            pass
        return s
    def boosted_rivals(r):
        c = 0.0
        try:
            for did, dv in alldr.items():
                st = dv.get('status', '')
                if st not in ('idle', 'relocating'):
                    continue
                bud = 1.0
                try:
                    bud = float(dbud.get(did, 1.0))
                except:
                    bud = 1.0
                if bud > 1.05:
                    dl = dv.get('location')
                    if dl is not None:
                        try:
                            if float(phi_ep.dist(dl, rp[r])) <= scale:
                                c += (bud - 1.0)
                        except:
                            pass
        except:
            pass
        return c
    scores = {}
    best = None
    best_r = None
    cur_val = None
    for r in sorted(cands):
        if r is None or r < 0 or r >= n:
            continue
        eff = cap_eff(r)
        pax = reg_pax.get(r, 0.0)
        cnt = reg_cnt.get(r, 0.0)
        multi = reg_multi.get(r, 0.0)
        prior = 0.0
        if od_in is not None:
            try:
                prior += float(od_in[r]) * 4.0
            except:
                pass
            try:
                prior += float(od_out[r]) * 1.0
            except:
                pass
        prior_units = (0.35 * dp_ramp + 0.65) * prior * od_orders * 0.02
        # A-mechanism: objective-priced supply-netted demand w/ seat+throughput weighting
        demand = eff + thru_frac * cnt + 0.3 * pax
        # B-mechanism: seat_frac lifts multi-party + OD-inflow structural prior
        demand = demand + seat_frac * (multi + 0.5 * prior_units) + prior_units
        if demand <= 0:
            demand = eff + 0.5 * prior_units
        demand = demand * dem_mult
        supp = nb_supply(r)
        net = demand - 0.6 * supp
        if fs > 0.5:
            net = net - 0.4 * boosted_rivals(r)
        t = scale
        if loc is not None:
            try:
                t = float(phi_ep.dist(loc, rp[r]))
            except:
                t = scale
        if not t >= 0:
            t = scale
        cruise_units = t / scale
        penalty = cruise_w * cruise_units / reach
        if empty_averse and cruise_units > 1.0:
            penalty = penalty * (1.0 + 0.4 * (cruise_units - 1.0))
        val = net - penalty
        if fs > 1.0 and fb < 1.0:
            val = val * max(fb, 0.3)
        try:
            valf = float(val)
        except:
            continue
        if not np.isfinite(valf):
            continue
        scores[r] = valf
        if r == cur:
            cur_val = valf
        if best is None or valf > best:
            best = valf
            best_r = r
    if best is None:
        return {}
    stay = cur_val if cur_val is not None else 0.0
    margin = 0.12 * (1.0 + 0.6 * cruise_w)
    if best_r == cur:
        return {}
    if best - stay <= margin or best <= margin:
        return {}
    out = {}
    for r, v in scores.items():
        if np.isfinite(v):
            out[int(r)] = float(v)
    return out
