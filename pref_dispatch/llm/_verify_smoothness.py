"""Offline check: the fleet-mix probe measures OBJECTIVE adaptation -- no LLM
call, no env rollout, no API key.

History (why this file changed shape). It used to test a §5.4 "smoothness"
number: the largest jump in the fleet's argmax-skill mix between adjacent points
of a REVENUE grid. That measure belonged to the v1 contract, where a combiner's
fourth argument was a ``Preference`` and adapting meant sliding one revenue/
service dial. The final version replaced that dial with ``w`` -- the episode's
reward function, a callable -- so there is no revenue grid to walk, and a
``Preference`` handed in where a callable is expected just makes every authored
program raise, which the adapter swallows into :data:`NO_PICK`. The old measure
therefore read ~0 for every program while still paying for grid x capture-size
probe calls. It is retired (:func:`combiner_eval._fleet_smoothness` is a 0.0
stub); the live question -- does the fleet move when the TARGET moves? -- is
:func:`combiner_eval.blindness_from_dists` over a grid of real ``w`` callables.

What this file proves, on hand-written combiners so it depends on no frozen
artifact:

1. ``fleet_pick_fractions(w)`` runs under the CURRENT capture contract
   (``(driver_obs, phi_ep, phi_step)`` triples) and returns a distribution.
2. A w-BLIND combiner (never reads ``w``) scores blindness 1.0 -- identical fleet
   mix under every objective.
3. A w-READING combiner scores blindness well below 1.0 on the same fleet and the
   same objective grid: its mix genuinely moves with the objective.
4. A gradual combiner (drivers cross at DIFFERENT objective points) lands
   somewhere between: the fleet slides rather than flipping as one block.
5. The retired smoothness stub returns 0.0 for every program, so nothing can
   silently re-enter fitness through it.

Run:  python -m pref_dispatch.llm._verify_smoothness
"""

from __future__ import annotations

from typing import Dict, List

from pref_dispatch.global_stats import EpisodeStats, GlobalStats
from pref_dispatch.llm.combiner_adapter import LLMCombiner, NO_PICK
from pref_dispatch.llm.combiner_eval import (
    _fleet_smoothness,
    blindness_from_dists,
)

SKILLS = ("revenue", "service", "enroute")


# --------------------------------------------------------------------------- #
# A synthetic fleet under the CURRENT (driver_obs, phi_ep, phi_step) contract.  #
# --------------------------------------------------------------------------- #
def _synthetic_fleet(n: int = 120):
    """``n`` idle cars spread along a pickup corridor, sharing one pending pool.

    The spread matters for check [4]: drivers sitting at different distances from
    the same fares have different break-even points, so a combiner that thresholds
    on a per-driver quantity flips them one at a time (a slide) instead of all at
    once (a block flip)."""
    pending = []
    for k in range(12):
        ox, oy = -73.99 + 0.001 * k, 40.75
        dx, dy = ox + 0.002 + 0.004 * k, oy + 0.001 * k   # ride length grows with k
        pending.append({
            "order_id": k,
            "origin": (ox, oy),
            "destination": (dx, dy),
            "num_passengers": 1,
            "waiting_time": 1.0,
            "origin_region": 0,
            "destination_region": 0,
        })

    phi_ep = EpisodeStats(
        num_drivers=n, driver_capacity=4, speed_kmh=35.0,
        region_centres=((-73.99, 40.75),), region_neighbours=((0,),),
        scale=6.0,
    )
    phi_step = GlobalStats(
        time=10.0, num_pending=len(pending), num_drivers=n, num_idle=n,
        total_free_capacity=4 * n, demand_pressure=0.5, mean_solo_time=6.0,
        region_demand=(float(len(pending)),), region_supply=(float(n),),
    )

    fleet = []
    for i in range(n):
        driver_obs = {
            "self": {
                "driver_id": i,
                # spread along the corridor -> different nearest fare per driver
                "location": (-73.99 + 0.001 * (i % 12), 40.75 + 0.0002 * (i // 12)),
                "status": "idle",
                "capacity": 4,
                "speed": None,
                "onboard_passengers": 0,
                "assigned_orders": [],
                "assigned_order_details": [],
                "committed_passengers": 0,
                "current_region": 0,
            },
            "pending_orders": pending,
        }
        fleet.append((driver_obs, phi_ep, phi_step))
    return fleet


def _capture(combiner: LLMCombiner, fleet) -> LLMCombiner:
    """Inject the sample directly -- the probe replays it, so no env is needed."""
    combiner.enable_capture(len(fleet))
    combiner._obs_samples = list(fleet)
    return combiner


# --------------------------------------------------------------------------- #
# Three scorers: w-blind, w-reading (block flip), w-reading (gradual).          #
# --------------------------------------------------------------------------- #
class _Scorer:
    """Minimal stand-in for a sandboxed program (same duck-typed contract)."""

    def __init__(self, fn):
        self.skill_scores = fn


def _blind():
    def skill_scores(driver_obs, phi_ep, phi_step, w):
        return {"revenue": 1.0, "service": 0.0, "enroute": 0.0}
    return _Scorer(skill_scores)


def _probe(w) -> float:
    """What the objective pays for one extra served order, on a fixed probe event.

    This is the self-derivation pattern the frozen combiners use: rather than
    being told what the objective wants, call it on a synthetic event and read
    the number back."""
    if w is None:
        return 0.0
    event = {
        "assigned_orders": [1], "completed_orders": [1], "picked_up_orders": [1],
        "assigned_solo_times": {1: 8.0}, "assigned_party_sizes": {1: 2},
        "assigned_service_times": {1: 10.0}, "extra_detour_time": 0.0,
        "distance_moved": 1.0, "is_empty_move": False, "is_idle_wait": False,
    }
    try:
        return float(w(event))
    except Exception:                       # a broken objective is not this test
        return 0.0


def _reading_block():
    """Whole fleet switches at one objective threshold (no per-driver spread)."""
    def skill_scores(driver_obs, phi_ep, phi_step, w):
        pay = _probe(w)
        return ({"revenue": 1.0, "service": 0.0, "enroute": 0.0} if pay >= 10.0
                else {"revenue": 0.0, "service": 1.0, "enroute": 0.0})
    return _Scorer(skill_scores)


def _reading_gradual():
    """Each driver crosses at its OWN threshold, so the fleet mix slides."""
    def skill_scores(driver_obs, phi_ep, phi_step, w):
        pay = _probe(w)
        # per-driver break-even spread over the corridor position
        bar = 2.0 + 1.5 * (driver_obs["self"]["driver_id"] % 12)
        return ({"revenue": 1.0, "service": 0.0, "enroute": 0.0} if pay >= bar
                else {"revenue": 0.0, "service": 1.0, "enroute": 0.0})
    return _Scorer(skill_scores)


# --------------------------------------------------------------------------- #
# The objective grid: real w callables, of visibly different generosity.        #
# --------------------------------------------------------------------------- #
def _objective_grid() -> List:
    def w_cheap(event):      # pays ~1 per served order
        return 1.0 * len(event.get("completed_orders", ()))

    def w_mid(event):        # pays ~8
        return 8.0 * len(event.get("completed_orders", ()))

    def w_rich(event):       # pays ~20
        return 20.0 * len(event.get("completed_orders", ()))

    return [w_cheap, w_mid, w_rich]


def _mixes(combiner: LLMCombiner, grid) -> List[Dict[str, float]]:
    return [combiner.fleet_pick_fractions(w) for w in grid]


def main() -> None:
    fleet = _synthetic_fleet()
    grid = _objective_grid()
    print(f"fleet={len(fleet)} idle cars, objective grid={len(grid)} w callables "
          f"(pay per served order: 1, 8, 20)\n")

    # [1] the probe runs under the current 3-tuple capture contract.
    blind = _capture(LLMCombiner(_blind(), SKILLS), fleet)
    blind_mixes = _mixes(blind, grid)
    assert all(m for m in blind_mixes), blind_mixes
    for m in blind_mixes:
        assert abs(sum(m.values()) - 1.0) < 1e-9, m
        assert NO_PICK not in m, ("a working scorer must not defer", m)
    print(f"[1] capture contract OK: fleet_pick_fractions ran on "
          f"(driver_obs, phi_ep, phi_step) triples; each mix sums to 1.")

    # [2] w-blind -> blindness 1.0 (identical mix under every objective).
    b_blind = blindness_from_dists(blind_mixes)
    assert abs(b_blind - 1.0) < 1e-9, (b_blind, blind_mixes)
    print(f"[2] w-blind OK: blindness={b_blind:.3f} (mix identical under all "
          f"{len(grid)} objectives: {blind_mixes[0]})")

    # [3] w-reading -> blindness well below 1.0 on the SAME fleet + grid.
    block = _capture(LLMCombiner(_reading_block(), SKILLS), fleet)
    block_mixes = _mixes(block, grid)
    b_block = blindness_from_dists(block_mixes)
    assert b_block < 0.5, (b_block, block_mixes)
    print(f"[3] w-reading OK: blindness={b_block:.3f} << blind's {b_blind:.3f}; "
          f"fleet mix moves {[dict(sorted(m.items())) for m in block_mixes]}")

    # [4] gradual: drivers cross at different objective points, so at least one
    #     grid point is a genuine MIX rather than an all-one-skill block.
    grad = _capture(LLMCombiner(_reading_gradual(), SKILLS), fleet)
    grad_mixes = _mixes(grad, grid)
    b_grad = blindness_from_dists(grad_mixes)
    assert b_grad < 1.0, (b_grad, grad_mixes)
    mixed_points = sum(1 for m in grad_mixes if max(m.values()) < 0.999)
    assert mixed_points >= 1, grad_mixes
    block_mixed = sum(1 for m in block_mixes if max(m.values()) < 0.999)
    assert mixed_points > block_mixed, (mixed_points, block_mixed)
    print(f"[4] gradual OK: blindness={b_grad:.3f}, {mixed_points}/{len(grid)} "
          f"grid points are a blended fleet vs {block_mixed}/{len(grid)} for the "
          f"block-flip combiner (it slides instead of snapping)")

    # [5] the retired smoothness term is a hard 0 for every program.
    for name, c in (("blind", blind), ("block", block), ("gradual", grad)):
        tvd = _fleet_smoothness(c, (0.0, 0.5, 1.0))
        assert tvd == 0.0, (name, tvd)
    print(f"[5] retired smoothness OK: _fleet_smoothness == 0.0 for all three "
          f"programs, so it cannot re-enter fitness.")

    print("\nALL fleet-mix / objective-adaptation offline checks passed "
          "(no API key used).")


if __name__ == "__main__":
    main()
