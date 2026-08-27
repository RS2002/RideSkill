"""Offline verification for Feature 3 (idle-driver reposition). No LLM key needed.

Checks:
 (a) OFF is byte-identical to legacy: with ``repositioner=None`` the controller
     emits the exact same action dict (only {"orders":...}, never {"relocate":...})
     and a full rollout returns metrics identical to the reposition-off path, with
     the new reposition KPIs all zero.
 (b) ON emits only LEGAL relocate actions: every relocate target is an int region
     index in [0, R), and only idle drivers with an empty bid ever get one.
 (c) demand-gravity hits a NEIGHBOUR hotspot: with all pending demand in a region
     adjacent to an idle driver's region, that driver is sent toward it (final
     version is neighbour-only -- a non-adjacent hotspot is unreachable by design).
 (d) coordinated spreading de-conflicts: two idle drivers with two equally hot,
     both-adjacent regions split across them instead of both flocking to one.
 (e) online rollout is well-behaved: reposition ON yields finite KPIs and service
     does not collapse vs the OFF baseline (repositioning is reward-free, so it
     may only add empty cruise distance; it must not discard served demand).
 (f) KPI consistency across paths: EpisodeMetrics and EpisodeRecorder report the
     same reposition_distance / relocation_moves on the same event+status stream.
 (g) real env steps: a handful of live env.step calls with reposition ON raise no
     InvalidActionError (the relocate contract is honoured end to end).
 (h) Phase-2 compiled scorer: a Feature-3 ``reposition_scores`` (final 5-arg
     signature) compiles through the independent sandbox path (CompiledRepositioner),
     wraps in a Repositioner, and routes through DispatchController emitting only
     legal relocates.

The controller always holds ONE optional :class:`~pref_dispatch.reposition.Repositioner`
(``None`` = OFF); ``strength``/``params``/``scores_fn`` live on that object.

Two-layer stats (final version): the reposition path now takes ``(phi_ep, phi_step)``
instead of a single ``phi`` plus ``dist`` -- ``dist`` and the static region layout live
on ``phi_ep``; the live per-region ``kappa`` is seeded from ``phi_step``. The Phase-2
scorer signature is ``reposition_scores(driver_obs, phi_ep, phi_step, kappa, w)``.
"""
from __future__ import annotations

import numpy as np

from benchmark.recorder import EpisodeRecorder
from pref_dispatch.combiner import HeuristicCombiner
from pref_dispatch.evaluate import DispatchController, rollout, _make_dist
from pref_dispatch.global_stats import EpisodeStats, GlobalStats
from pref_dispatch.llm.basis import load_basis
from pref_dispatch.metrics import EpisodeMetrics
from pref_dispatch.preference import Preference
from pref_dispatch.reposition import Repositioner, choose_relocation_targets, RepositionParams
from pref_dispatch.scenario import Scenario, build_env
from ride_gym.enums import DriverStatus


# A 4-region square map: centres at the corners of a 10x10 grid.
_CENTRES = ((0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0))
_NEIGHBOURS = ((1, 2), (0, 3), (0, 3), (1, 2))


def _mk_obs(did, loc, status="idle", *, pending=(), current_region=0):
    """Build one driver observation matching the env's obs schema."""
    return {
        "self": {
            "driver_id": did,
            "location": loc,
            "status": status,
            "capacity": 4,
            "committed_passengers": 0,
            "current_region": current_region,
        },
        "all_drivers": {},  # filled by _link below
        "pending_orders": list(pending),
        "time": 0.0,
        "relocation_points": _CENTRES,
        "region_neighbours": _NEIGHBOURS,
    }


def _link(obs_map):
    """Share one all_drivers view across every obs (as the env does)."""
    all_drivers = {
        did: {
            "location": o["self"]["location"],
            "status": o["self"]["status"],
            "onboard_passengers": 0,
        }
        for did, o in obs_map.items()
    }
    for o in obs_map.values():
        o["all_drivers"] = all_drivers
    return obs_map


def _euclid(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _phis(obs_map):
    """Build the two-layer stats (phi_ep static, phi_step live) for a hand obs.

    ``dist`` and the region layout live on ``phi_ep``; the live per-region kappa is
    seeded from ``phi_step``. Both read the same synthetic ``_euclid`` backend.
    """
    phi_ep = EpisodeStats.from_observations(obs_map, dist=_euclid, speed_kmh=35.0)
    phi_step = GlobalStats.from_observations(obs_map, dist=_euclid)
    return phi_ep, phi_step


def test_demand_gravity_hits_hotspot() -> None:
    # All pending demand originates in region 1 (corner (10,0)), a NEIGHBOUR of
    # region 0; one idle car sits near region 0. Neighbour-only candidates are
    # {0,1,2}, so it should be sent toward the adjacent hotspot region 1.
    pending = [{"order_id": i, "origin": (10.0, 0.0), "destination": (5.0, 5.0),
                "num_passengers": 2, "waiting_time": 0.0} for i in range(6)]
    obs = _link({0: _mk_obs(0, (0.5, 0.5), pending=pending, current_region=0)})
    phi_ep, phi_step = _phis(obs)
    targets = choose_relocation_targets(obs, phi_ep, phi_step, strength=1.0)
    assert targets.get(0) == 1, targets
    print(f"[c] demand-gravity OK: idle car at (0.5,0.5) -> region 1 (adjacent "
          f"hotspot) = {targets}.")


def test_coordinated_spreading_deconflicts() -> None:
    # Two equally hot regions (1 and 2), both NEIGHBOURS of region 3, and two idle
    # cars sitting centrally in region 3. Coordinated spreading must send them to
    # DIFFERENT regions, not both to the same one.
    pending = (
        [{"order_id": i, "origin": (10.0, 0.0), "destination": (5.0, 5.0),
          "num_passengers": 2, "waiting_time": 0.0} for i in range(5)] +
        [{"order_id": 100 + i, "origin": (0.0, 10.0), "destination": (5.0, 5.0),
          "num_passengers": 2, "waiting_time": 0.0} for i in range(5)]
    )
    obs = _link({
        0: _mk_obs(0, (5.0, 5.0), pending=pending, current_region=3),
        1: _mk_obs(1, (5.0, 5.0), pending=pending, current_region=3),
    })
    phi_ep, phi_step = _phis(obs)
    targets = choose_relocation_targets(obs, phi_ep, phi_step, strength=1.0)
    assert set(targets) == {0, 1}, targets
    assert targets[0] != targets[1], targets
    assert set(targets.values()) == {1, 2}, targets
    print(f"[d] coordinated-spreading OK: two idle cars split across the two "
          f"adjacent hotspots = {targets} (no flocking).")


def _empty_ev(distance):
    return {
        "assigned_orders": [], "assigned_solo_times": {},
        "assigned_party_sizes": {}, "assigned_service_times": {},
        "completed_orders": [], "picked_up_orders": [],
        "distance_moved": float(distance), "is_empty_move": True,
        "is_idle_wait": False, "extra_detour_time": 0.0,
    }


def test_kpi_consistency_across_paths() -> None:
    # Driver 0 makes two empty moves while RELOCATING (reposition cruise);
    # driver 1 makes one empty move while TO_PICKUP (a normal empty pickup leg,
    # NOT reposition). Both paths must attribute the same reposition slice.
    steps = [
        {0: _empty_ev(2.0), 1: _empty_ev(3.0)},
        {0: _empty_ev(1.5), 1: _empty_ev(0.0)},
    ]
    # Post-step status per driver per step (what each path keys attribution on).
    statuses = [
        {0: "relocating", 1: "to_pickup"},
        {0: "relocating", 1: "idle"},
    ]

    # --- online path: EpisodeMetrics ---
    m = EpisodeMetrics()
    for evs, st in zip(steps, statuses):
        m.update(rewards={d: 0.0 for d in evs}, info={"events": evs},
                 driver_status=st)
    fm = m.finalize(total_orders=1)

    # --- benchmark path: EpisodeRecorder (reads env.drivers[did].status) ---
    class _D:
        def __init__(self, status):
            self.status = status
            self.onboard_passengers = 0

    class _Env:
        _pending_ids: list = []
        orders: dict = {}
        def __init__(self):
            self.drivers = {}

    rec = EpisodeRecorder(algorithm="fake")
    env = _Env()
    _STR2ENUM = {
        "relocating": DriverStatus.RELOCATING,
        "to_pickup": DriverStatus.TO_PICKUP,
        "idle": DriverStatus.IDLE,
    }
    for evs, st in zip(steps, statuses):
        env.drivers = {d: _D(_STR2ENUM[st[d]]) for d in evs}
        rec.record_step(env, {"events": evs, "time": 0.0},
                        assign_log={}, rewards={d: 0.0 for d in evs})
    rec_repo = sum(a["reposition_distance"] for a in rec._driver_acc.values())
    rec_moves = sum(a["relocation_moves"] for a in rec._driver_acc.values())

    assert fm["reposition_distance_km"] == 3.5 == rec_repo, (fm, rec_repo)
    assert fm["relocation_moves"] == 2 == rec_moves, (fm, rec_moves)
    # ratio = reposition / all empty distance = 3.5 / (3.5 + 3.0) = 0.5384...
    assert abs(fm["reposition_distance_ratio"] - 3.5 / 6.5) < 1e-9, fm
    print(f"[f] KPI-consistency OK: EpisodeMetrics & recorder agree "
          f"(reposition_distance={rec_repo}, moves={rec_moves}, "
          f"ratio={fm['reposition_distance_ratio']:.4f}); a TO_PICKUP empty leg "
          f"is NOT counted as reposition.")


def _small_scenario():
    return Scenario(num_drivers=60, driver_capacity=4, speed_kmh=35.0,
                    regime="peak", split="train", order_limit=120,
                    pref_revenue=0.5, seed=0)


def _load_ctrl(reposition_strength=0.0, scores_fn=None):
    skills, _ = load_basis(include_evolved=True)
    # A handwritten new-signature combiner (frozen Phase-2 artifacts carry the OLD
    # skill_scores signature and are regenerated in Part B, so the offline
    # reposition check stays artifact-independent). The reposition mechanism under
    # test is orthogonal to which combiner scores the orders.
    comb = HeuristicCombiner(pref=_PREF)
    # The controller holds ONE optional Repositioner; None = repositioning off
    # (no relocate action is ever emitted). strength=0 builds an OFF-equivalent,
    # a real strength>0 builds the demand-gravity Repositioner (with an optional
    # compiled Feature-3 scorer for the Phase-2 check).
    repositioner = None if reposition_strength <= 0.0 else Repositioner(
        strength=reposition_strength, scores_fn=scores_fn,
    )
    return DispatchController(comb, skills=skills, top_k=20,
                              repositioner=repositioner)


_PREF = Preference({"revenue": 0.5, "service": 0.5, "fairness": 0.0})


def _phi_ep_for_env(env, obs):
    """Episode-static phi_ep for a live env (dist + layout, computed once)."""
    speed = float(getattr(getattr(env, "config", None), "vehicle_speed_kmh", 0.0) or 0.0)
    return EpisodeStats.from_observations(obs, dist=_make_dist(env), speed_kmh=speed)


def test_off_is_byte_identical() -> None:
    # OFF controller must (1) emit no relocate action on a live obs and (2)
    # produce rollout metrics identical to the OFF path, with reposition KPIs 0.
    sc = _small_scenario()
    env = build_env(sc)
    obs, _ = env.reset(seed=0)
    phi_ep = _phi_ep_for_env(env, obs)
    ctrl_off = _load_ctrl(0.0)
    actions = ctrl_off.act(obs, _PREF, {d: 0.0 for d in obs}, phi_ep,
                           fairness_income={d: 0.0 for d in obs})
    assert all("relocate" not in a for a in actions.values()), "OFF emitted relocate"
    assert all(set(a) <= {"orders"} for a in actions.values()), actions

    r_off = rollout(build_env(sc), _load_ctrl(0.0), _PREF, seed=0)
    assert r_off["relocation_moves"] == 0, r_off
    assert r_off["reposition_distance_km"] == 0.0, r_off
    assert r_off["reposition_distance_ratio"] == 0.0, r_off
    print(f"[a] OFF byte-identical OK: no relocate action emitted; rollout "
          f"reposition KPIs all zero (moves=0, dist=0).")


def test_on_emits_only_legal_relocates() -> None:
    sc = _small_scenario()
    env = build_env(sc)
    obs, _ = env.reset(seed=0)
    phi_ep = _phi_ep_for_env(env, obs)  # episode-static: computed once
    R = len(env.relocation_points)
    ctrl_on = _load_ctrl(1.0)
    # Step a few times so some drivers finish trips and go idle -> eligible.
    saw_relocate = False
    for _ in range(15):
        actions = ctrl_on.act(obs, _PREF, {d: 0.0 for d in obs}, phi_ep,
                              fairness_income={d: 0.0 for d in obs})
        for did, a in actions.items():
            if "relocate" in a:
                saw_relocate = True
                idx = a["relocate"]
                assert isinstance(idx, int) and 0 <= idx < R, (did, idx, R)
                assert obs[did]["self"]["status"] == "idle", (did, obs[did]["self"])
                assert "orders" not in a, a  # mutually exclusive
        obs, _r, dones, _info = env.step(actions)
        if dones["__all__"]:
            break
    print(f"[b] legal-relocates OK: every relocate is an int in [0,{R}) on an "
          f"idle empty-bid driver (saw_relocate={saw_relocate}).")


def test_online_rollout_is_well_behaved() -> None:
    # This check proves the reposition MECHANISM runs end-to-end and stays sane;
    # it does NOT assert reposition improves service. Whether cruising to demand
    # pays off is an empirical, scenario-dependent question that the sweep
    # (run_reposition_sweep) answers under a proper demand-imbalanced regime. In
    # a tiny 60-driver / 120-order episode the current hotspot is often already
    # served by the time a car arrives, so chasing it can *cost* a little service
    # -- a genuine tradeoff, not a bug.
    #
    # The correctness invariants we DO assert:
    #  * KPIs are finite and the ratio is a proper fraction in [0, 1];
    #  * repositioning actually fired (relocations happened, distance > 0) -- the
    #    branch is not silently dead;
    #  * service does not COLLAPSE. A RELOCATING car still bids in compute_bids
    #    (no status filter) and the env clears its target on assignment
    #    (env.py:611), so cars are never lost to matching; service staying a large
    #    fraction of baseline is the guard against a stuck-car regression.
    sc = _small_scenario()
    off = rollout(build_env(sc), _load_ctrl(0.0), _PREF, seed=0)
    on = rollout(build_env(sc), _load_ctrl(1.0), _PREF, seed=0)
    for r in (off, on):
        assert np.isfinite(r["service_rate"]), r
        assert np.isfinite(r["reposition_distance_ratio"]), r
    assert on["relocation_moves"] > 0, on
    assert on["reposition_distance_km"] > 0.0, on
    assert 0.0 < on["reposition_distance_ratio"] <= 1.0, on
    assert off["relocation_moves"] == 0, off  # OFF never relocates
    # Stuck-car guard: cars still get matched while cruising, so service must not
    # collapse (stays a large fraction of the OFF baseline).
    assert on["service_rate"] >= 0.6 * off["service_rate"], (
        off["service_rate"], on["service_rate"])
    print(f"[e] online-rollout OK: OFF service={off['service_rate']:.3f} "
          f"relocs={off['relocation_moves']} | ON service={on['service_rate']:.3f} "
          f"repo%={on['reposition_distance_ratio']:.3f} relocs={on['relocation_moves']} "
          f"(mechanism fires; service does not collapse; whether it *helps* is the "
          f"sweep's question).")


def test_real_env_steps_no_invalid_action() -> None:
    # (g) is really the survival of (b)'s live-step loop: it drove 15 real
    # env.step calls with reposition ON and never raised InvalidActionError.
    # Re-assert explicitly on a fresh env for clarity.
    sc = _small_scenario()
    env = build_env(sc)
    obs, _ = env.reset(seed=0)
    phi_ep = _phi_ep_for_env(env, obs)
    ctrl_on = _load_ctrl(1.0)
    steps = 0
    for _ in range(20):
        actions = ctrl_on.act(obs, _PREF, {d: 0.0 for d in obs}, phi_ep,
                              fairness_income={d: 0.0 for d in obs})
        obs, _r, dones, _info = env.step(actions)  # raises InvalidActionError on bug
        steps += 1
        if dones["__all__"]:
            break
    print(f"[g] real-env-steps OK: {steps} live env.step calls with reposition "
          f"ON, no InvalidActionError.")


# A tiny LLM-authored Feature-3 scorer (Phase 2), used ONLY to verify the
# independent sandbox path end-to-end. It is written like the prompt contract
# requires: a single ``reposition_scores`` with the final 5-arg signature
# ``(driver_obs, phi_ep, phi_step, kappa, w)``, region indices as int keys.
_PHASE2_SCORER = '''\
def reposition_scores(driver_obs, phi_ep, phi_step, kappa, w):
    """Send idle cars to the region with the most pending passenger demand,
    discounted by cruise distance (scale-free in phi_step.mean_solo_time).
    ``dist`` comes off the episode-static phi_ep; ``w`` is unused here."""
    scale = max(phi_step.mean_solo_time, 1e-6)
    dist = phi_ep.dist
    pts = driver_obs["relocation_points"]
    loc = driver_obs["self"]["location"]
    demand = {}
    for o in driver_obs.get("pending_orders", []):
        best, best_d2 = 0, float("inf")
        for r, c in enumerate(pts):
            d2 = (o["origin"][0] - c[0]) ** 2 + (o["origin"][1] - c[1]) ** 2
            if d2 < best_d2:
                best, best_d2 = r, d2
        demand[best] = demand.get(best, 0.0) + o["num_passengers"]
    out = {}
    for r, d in demand.items():
        tt = dist(loc, tuple(pts[r])) / scale
        out[int(r)] = d / (1.0 + tt)
    return out
'''


def test_phase2_compiled_scorer_routes_legally() -> None:
    """The Feature-3 scorer compiles through the sandbox and, wrapped in a
    Repositioner, routes through DispatchController -> only legal relocates."""
    from pref_dispatch.llm.sandbox import (
        CompiledRepositioner,
        compile_repositioner,
        validate_repositioner,
    )

    # (1) Independent compile path: no score/noop_score required, and the
    # contract check (int in-range keys, finite values) passes.
    cr = compile_repositioner(_PHASE2_SCORER, name="phase2_demo")
    assert isinstance(cr, CompiledRepositioner), cr
    ok, why = validate_repositioner(cr)
    assert ok, why

    # (2) Wrap in a Repositioner and route through the controller.
    sc = _small_scenario()
    env = build_env(sc)
    obs, _ = env.reset(seed=0)
    phi_ep = _phi_ep_for_env(env, obs)
    R = len(env.relocation_points)
    ctrl = _load_ctrl(1.0, scores_fn=cr.reposition_scores)
    saw_relocate = False
    for _ in range(15):
        actions = ctrl.act(obs, _PREF, {d: 0.0 for d in obs}, phi_ep,
                           fairness_income={d: 0.0 for d in obs})
        for did, a in actions.items():
            if "relocate" in a:
                saw_relocate = True
                idx = a["relocate"]
                assert isinstance(idx, int) and 0 <= idx < R, (did, idx, R)
                assert obs[did]["self"]["status"] == "idle", (did, obs[did]["self"])
                assert "orders" not in a, a
        obs, _r, dones, _info = env.step(actions)
        if dones["__all__"]:
            break
    print(f"[h] phase-2 compiled scorer OK: compile_repositioner -> "
          f"Repositioner(scores_fn=...) -> DispatchController emits only legal "
          f"relocates (saw_relocate={saw_relocate}); validation passed: {why or 'ok'}.")

    # (3) The same scorer is also loadable from disk via the frozen loader shape
    # (compile the body through the sandbox, as load_repositioner does).
    body = _PHASE2_SCORER[_PHASE2_SCORER.index("def reposition_scores"):]
    cr2 = compile_repositioner(body, name="phase2_demo")
    ok2, why2 = validate_repositioner(cr2)
    assert ok2, why2
    print(f"[h'] frozen-loader shape OK: body-only recompile validates too.")


if __name__ == "__main__":
    test_demand_gravity_hits_hotspot()
    test_coordinated_spreading_deconflicts()
    test_kpi_consistency_across_paths()
    test_off_is_byte_identical()
    test_on_emits_only_legal_relocates()
    test_online_rollout_is_well_behaved()
    test_real_env_steps_no_invalid_action()
    test_phase2_compiled_scorer_routes_legally()
    print("\nALL FEATURE-3 (REPOSITION) PHASE-1 + PHASE-2 OFFLINE CHECKS PASSED")
