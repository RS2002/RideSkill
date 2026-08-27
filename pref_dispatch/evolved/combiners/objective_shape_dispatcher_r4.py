# Phase-2 CHECKPOINT (not a frozen artifact): objective_shape_dispatcher at ['7']
# Written 2026-08-26 15:36:12. Inert until moved into pref_dispatch/evolved/ by hand.

def skill_scores(driver_obs, phi_ep, phi_step, w):
    self_obs = driver_obs["self"]
    details = self_obs.get("assigned_order_details", []) or []
    etas = [d["eta"] for d in details if d.get("eta") is not None]
    min_slack = min(etas) if etas else None
    cap = float(self_obs.get("capacity", 1) or 1)
    onboard = float(self_obs.get("committed_passengers", 0) or 0)
    m = float(phi_step.mean_solo_time) or float(phi_ep.scale) or 1.0
    dp = float(phi_step.demand_pressure)
    pend = driver_obs.get("pending_orders", []) or []
    loc = self_obs.get("location", (0.0, 0.0))

    # ---- deadline-pressed: protect onboard regardless of objective ----
    if min_slack is not None and min_slack <= 0.7 * m:
        return {"enroute": 1.0, "slack_budget_gate_ranked": 0.5,
                "freeness_option_patience": 0.3, "service": 0.2}

    loaded = min_slack is not None

    # ---- detect objective shape via guarded delta-probes ----
    # default signals (used when w is None or probing fails)
    sig = {"compl": 0.0, "seat": 0.0, "len": 0.0, "vol": 0.0,
           "empty": 0.0, "idle": 0.0, "det_new": 0.0, "det_on": 0.0,
           "pick": 0.0, "disp": 0.0}
    have_sig = False
    if w is not None:
        try:
            def base(**kw):
                e = {
                    "assigned_orders": [], "assigned_party_sizes": {},
                    "assigned_dispatch_wait": {}, "assigned_pickup_times": {},
                    "assigned_solo_times": {}, "assigned_service_times": {},
                    "assigned_detour_times": {}, "completed_orders": [],
                    "picked_up_orders": [], "distance_moved": 0.0,
                    "time_moved": 0.0, "is_empty_move": False,
                    "is_idle_wait": False, "extra_detour_time": 0.0,
                }
                for k, v in kw.items():
                    e[k] = v
                return float(w(e))
            p = 1
            sw = 0.6 * m
            # baseline single new order, short solo, short pickup
            short = base(assigned_orders=[p], assigned_party_sizes={p: 1},
                         assigned_dispatch_wait={p: sw},
                         assigned_pickup_times={p: 0.5 * m},
                         assigned_solo_times={p: 0.5 * m},
                         assigned_service_times={p: sw + 1.0 * m})
            # long solo ride (length/revenue signal)
            long = base(assigned_orders=[p], assigned_party_sizes={p: 1},
                        assigned_dispatch_wait={p: sw},
                        assigned_pickup_times={p: 0.5 * m},
                        assigned_solo_times={p: 3.0 * m},
                        assigned_service_times={p: sw + 3.5 * m})
            sig["len"] = long - short
            # completion delta
            compl = base(assigned_orders=[p], assigned_party_sizes={p: 1},
                         assigned_solo_times={p: 0.5 * m},
                         assigned_service_times={p: sw + 1.0 * m},
                         completed_orders=[p])
            sig["compl"] = compl - short
            # seating delta (party 2 vs 1)
            party2 = base(assigned_orders=[p], assigned_party_sizes={p: 2},
                          assigned_dispatch_wait={p: sw},
                          assigned_pickup_times={p: 0.5 * m},
                          assigned_solo_times={p: 0.5 * m},
                          assigned_service_times={p: sw + 1.0 * m})
            sig["seat"] = party2 - short
            # volume delta (two new orders vs one)
            q = 2
            vol2 = base(assigned_orders=[p, q],
                        assigned_party_sizes={p: 1, q: 1},
                        assigned_solo_times={p: 0.5 * m, q: 0.5 * m},
                        assigned_service_times={p: sw + 1.0 * m,
                                                q: sw + 1.0 * m})
            sig["vol"] = (vol2 - short) - short  # marginal of 2nd order vs 1st
            # long pickup vs short pickup
            longpick = base(assigned_orders=[p], assigned_party_sizes={p: 1},
                            assigned_dispatch_wait={p: sw},
                            assigned_pickup_times={p: 3.0 * m},
                            assigned_solo_times={p: 0.5 * m},
                            assigned_service_times={p: sw + 3.5 * m})
            sig["pick"] = longpick - short
            # dispatch-wait price
            nodw = base(assigned_orders=[p], assigned_party_sizes={p: 1},
                        assigned_dispatch_wait={p: 0.0},
                        assigned_pickup_times={p: 0.5 * m},
                        assigned_solo_times={p: 0.5 * m},
                        assigned_service_times={p: 1.0 * m})
            sig["disp"] = short - nodw
            # bundling: new-order detour and onboard detour
            bund_no = base(assigned_orders=[p], picked_up_orders=[q],
                           assigned_party_sizes={p: 1},
                           assigned_solo_times={p: 0.5 * m},
                           assigned_service_times={p: sw + 1.0 * m},
                           extra_detour_time=0.0)
            bund_det = base(assigned_orders=[p], picked_up_orders=[q],
                            assigned_party_sizes={p: 1},
                            assigned_solo_times={p: 0.5 * m},
                            assigned_service_times={p: sw + 2.0 * m},
                            assigned_detour_times={p: 1.5 * m},
                            extra_detour_time=1.5 * m)
            sig["det_new"] = bund_det - bund_no
            bund_on = base(assigned_orders=[p], picked_up_orders=[q],
                           assigned_party_sizes={p: 1},
                           assigned_solo_times={p: 0.5 * m},
                           assigned_service_times={p: sw + 1.0 * m},
                           assigned_detour_times={q: 1.5 * m},
                           extra_detour_time=1.5 * m)
            sig["det_on"] = bund_on - bund_no
            # empty-move aversion
            emov = base(is_empty_move=True, distance_moved=1.0 * m,
                        time_moved=1.0 * m)
            sig["empty"] = emov - base()
            iwait = base(is_idle_wait=True)
            sig["idle"] = iwait - base()
            have_sig = True
        except Exception:
            have_sig = False

    # ---- build blend from detected shape ----
    out = {}
    if have_sig:
        # normalise magnitudes for comparison
        scale = (abs(sig["compl"]) + abs(sig["seat"]) + abs(sig["len"]) +
                 abs(sig["vol"]) + abs(sig["empty"]) + abs(sig["idle"]) +
                 abs(sig["pick"]) + 1e-9)
        compl = sig["compl"] / scale
        seat = sig["seat"] / scale
        length = sig["len"] / scale
        vol = sig["vol"] / scale
        empty_av = -sig["empty"] / scale  # positive => empty moves punished
        idle_av = -sig["idle"] / scale     # positive => idle punished
        det_pen = -(sig["det_new"] + sig["det_on"]) / scale

        # start weights
        out = {
            "service": 0.15, "slack_budget_gate_ranked": 0.15,
            "near_short_efficiency": 0.15, "seat_first_marginal_load": 0.1,
            "revenue": 0.1, "gowin_ratio_gate": 0.1,
            "freeness_option_patience": 0.1,
            "dualfree_ratio_lastseat_topoff": 0.05,
            "corridor_gated_starvation_seeding_pool": 0.05,
            "enroute": 0.05,
        }
        # completion-gated -> reliable fast low-detour serve
        if compl > 0.12:
            out["slack_budget_gate_ranked"] += 0.8 * compl
            out["service"] += 0.6 * compl
            out["near_short_efficiency"] += 0.4 * compl
        # seating/pooling -> fill seats
        if seat > 0.12:
            out["freeness_option_patience"] += 0.7 * seat
            out["dualfree_ratio_lastseat_topoff"] += 0.7 * seat
            out["corridor_gated_starvation_seeding_pool"] += 0.5 * seat
            out["seat_first_marginal_load"] += 0.4 * seat
        # length/revenue -> long fares
        if length > 0.12:
            out["revenue"] += 0.8 * length
            out["gowin_ratio_gate"] += 0.7 * length
        # assignment volume -> broad coverage
        if vol > 0.12:
            out["seat_first_marginal_load"] += 0.7 * vol
            out["near_short_efficiency"] += 0.6 * vol
        # empty/idle aversion -> never sit, take feasible work
        if empty_av > 0.1 or idle_av > 0.1:
            av = max(empty_av, idle_av)
            out["near_short_efficiency"] += 0.7 * av
            out["seat_first_marginal_load"] += 0.5 * av
            out["service"] = out.get("service", 0.0) - 0.2 * av
        # detour penalty -> protect onboard, tighten pooling gates
        if det_pen > 0.1:
            out["slack_budget_gate_ranked"] += 0.5 * det_pen
            out["enroute"] += 0.3 * det_pen
            out["freeness_option_patience"] += 0.3 * det_pen

        # loaded-with-slack car: bias toward cheap top-ups of spare seats
        if loaded and onboard < cap:
            out["dualfree_ratio_lastseat_topoff"] = out.get(
                "dualfree_ratio_lastseat_topoff", 0.0) + 0.4
            out["freeness_option_patience"] = out.get(
                "freeness_option_patience", 0.0) + 0.3
            out["enroute"] = out.get("enroute", 0.0) + 0.2
    else:
        # ---- w is None: demand-pressure balanced fallback ----
        cover = min(1.0, 0.4 + 0.4 * dp)  # scarce -> broad coverage
        out = {
            "service": 1.0 - 0.5 * cover,
            "near_short_efficiency": cover,
            "slack_budget_gate_ranked": 0.5,
            "seat_first_marginal_load": 0.3 * cover,
            "enroute": 0.2,
        }
        if loaded and onboard < cap:
            out["dualfree_ratio_lastseat_topoff"] = 0.4
            out["freeness_option_patience"] = 0.3

    # ensure at least one positive skill
    if not any(v > 0 for v in out.values()):
        out = {"service": 1.0, "near_short_efficiency": 0.5, "enroute": 0.2}
    # sanitise
    clean = {}
    for k, v in out.items():
        fv = float(v)
        if fv != fv or fv in (float("inf"), float("-inf")):
            fv = 0.0
        clean[k] = fv
    return clean
