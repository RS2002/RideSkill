"""Offline check of v6 item 8: top-k skill blending with per-skill normalization.

Four claims, no API key and no real map needed:

(a) ``blend_k=1`` is the OLD one-hot path, exactly. A one-hot weight dict routed
    through :func:`_candidate_scores` returns the skill's RAW scores, and the
    whole ``compute_bids`` output is identical to what the pre-v6 weighted-sum
    code produced (reproduced here by ``_blended_score``, which is untouched).
(b) The bug item 8 fixes is real: with raw scores a 50/50 blend of a
    dollars-scale skill and a 0..1-scale skill is decided ~entirely by the
    dollars skill; after per-skill standardization both actually count.
(c) An order any participating skill declares infeasible is dropped, even when
    another skill scores it highly (the old weighted-sum test let a small-weight
    floor slip through).
(d) The combiner-side top-k weights depend only on the RANKING and SEPARATION of
    the LLM's scores, not their magnitude: scaling every score by 1000 gives the
    same weights.
"""

from __future__ import annotations

import numpy as np

from pref_dispatch.matching import (
    _active_skills,
    _blended_score,
    _candidate_scores,
    compute_bids,
)


# --------------------------------------------------------------------------- #
# Minimal stand-ins (no env, no map).                                          #
# --------------------------------------------------------------------------- #
class _Skill:
    """Scores an order by a named field, times a scale."""

    def __init__(self, name, field, scale=1.0, floor_ids=()):
        self.name = name
        self.field = field
        self.scale = scale
        self.floor_ids = set(floor_ids)

    def score(self, driver_obs, order, phi_ep, phi_step):
        if order["order_id"] in self.floor_ids:
            return -1e9
        return self.scale * float(order[self.field])

    def noop_score(self, driver_obs, phi_ep, phi_step):
        return 0.0


class _Fixed:
    """Combiner returning a fixed weight dict for every driver."""

    def __init__(self, weights):
        self.weights = dict(weights)

    def weights_for(self, driver_obs, phi_ep, phi_step, w=None):
        return dict(self.weights)

    def classify(self, driver_obs, phi_ep, phi_step):
        return max(self.weights, key=self.weights.get)


def _orders():
    # fare in dollars (tens), urgency in 0..1 -- the two scales that broke the
    # old raw-sum blend. Order 2 is the one urgency likes and fare does not.
    return [
        {"order_id": 0, "origin": (0.0, 0.0), "fare": 40.0, "urgency": 0.10},
        {"order_id": 1, "origin": (1.0, 0.0), "fare": 30.0, "urgency": 0.20},
        {"order_id": 2, "origin": (2.0, 0.0), "fare": 28.0, "urgency": 0.95},
    ]


def _obs(n=2):
    orders = _orders()
    return {
        d: {"self": {"driver_id": d, "location": (0.0, 0.0)}, "pending_orders": orders}
        for d in range(n)
    }


PHI_EP = object()
PHI_STEP = object()


# --------------------------------------------------------------------------- #
def test_blend_k1_is_raw_one_hot() -> None:
    fare = _Skill("fare", "fare")
    urg = _Skill("urgency", "urgency")
    skills = {"fare": fare, "urgency": urg}
    orders = _orders()

    active = _active_skills(skills, {"fare": 1.0}, blend_k=1)
    assert active == [("fare", 1.0)], active
    scores, feasible = _candidate_scores(
        skills, active, {}, orders, PHI_EP, PHI_STEP, -1e8
    )
    old = [
        _blended_score(skills, {"fare": 1.0}, {}, o, PHI_EP, PHI_STEP)
        for o in orders
    ] + [_blended_score(skills, {"fare": 1.0}, {}, None, PHI_EP, PHI_STEP, noop=True)]
    assert np.allclose(scores, old), (scores, old)
    assert feasible.all()

    # ...and a multi-skill weight dict truncated to k=1 is the top weight alone.
    active = _active_skills(skills, {"fare": 0.4, "urgency": 0.6}, blend_k=1)
    assert active == [("urgency", 1.0)], active
    print(f"[a] blend_k=1 parity OK: raw scores {list(scores)} == pre-v6 blend")


def test_normalization_makes_weights_real() -> None:
    skills = {"fare": _Skill("fare", "fare"), "urgency": _Skill("urgency", "urgency")}
    orders = _orders()
    w = {"fare": 0.5, "urgency": 0.5}

    raw = np.array(
        [_blended_score(skills, w, {}, o, PHI_EP, PHI_STEP) for o in orders]
    )
    # Old behaviour: fare's ranking wins outright (order 0), urgency invisible.
    assert int(raw.argmax()) == 0, raw
    # How much each skill moved the raw score between order 0 and order 2 (the
    # pair the two skills disagree about): fare says +6.00, urgency says -0.43,
    # so urgency is outvoted ~14:1 despite carrying half the weight.
    fare_move = (orders[0]["fare"] - orders[2]["fare"]) * 0.5
    urg_move = (orders[0]["urgency"] - orders[2]["urgency"]) * 0.5
    assert abs(fare_move) > 10 * abs(urg_move), (fare_move, urg_move)

    new, _feas = _candidate_scores(skills, [("fare", 0.5), ("urgency", 0.5)],
                                   {}, orders, PHI_EP, PHI_STEP, -1e8)
    # After standardization the high-urgency order 2 wins the blend.
    assert int(np.argmax(new[:3])) == 2, new
    print(f"[b] normalization OK: on raw scores fare moves the 0-vs-2 decision "
          f"by {fare_move:+.2f} and urgency by {urg_move:+.2f} at equal weight "
          f"-> order 0; standardized blend picks order 2")


def test_any_skill_floor_drops_the_order() -> None:
    skills = {
        "fare": _Skill("fare", "fare"),
        "cap": _Skill("cap", "urgency", floor_ids={0}),  # order 0 infeasible
    }
    orders = _orders()
    # Old test: a 0.01 weight on the flooring skill left the blend above the
    # floor, so the infeasible order survived.
    old = _blended_score(skills, {"fare": 0.99, "cap": 0.01}, {}, orders[0],
                         PHI_EP, PHI_STEP)
    assert old > -1e8, old
    _s, feasible = _candidate_scores(
        skills, [("fare", 0.99), ("cap", 0.01)], {}, orders, PHI_EP, PHI_STEP, -1e8
    )
    assert not feasible[0] and feasible[1] and feasible[2], feasible
    assert feasible[-1], "the no-op must always stay available"
    print(f"[c] feasibility OK: old blended score {old:.3g} > floor (would have "
          f"bid an infeasible order); new mask drops it")


def test_compute_bids_end_to_end() -> None:
    skills = {"fare": _Skill("fare", "fare"), "urgency": _Skill("urgency", "urgency")}
    obs = _obs(2)
    budgets = {d: 1.0 for d in obs}

    one_hot, _ = compute_bids(
        observations=obs, skills=skills, combiner=_Fixed({"fare": 1.0}),
        phi_ep=PHI_EP, phi_step=PHI_STEP, budgets=budgets,
        temperature=1.0, top_k=0, blend_k=1,
    )
    mixed, _ = compute_bids(
        observations=obs, skills=skills,
        combiner=_Fixed({"fare": 0.5, "urgency": 0.5}),
        phi_ep=PHI_EP, phi_step=PHI_STEP, budgets=budgets,
        temperature=1.0, top_k=0, blend_k=2,
    )
    flat = [o for v in one_hot.values() for o in v]
    assert len(flat) == len(set(flat)), one_hot     # conflict-free
    flat_m = [o for v in mixed.values() for o in v]
    assert len(flat_m) == len(set(flat_m)), mixed
    assert one_hot != mixed, "blending changed nothing -- normalization inert?"
    print(f"[e] compute_bids OK: one-hot {dict(one_hot)} vs blended {dict(mixed)}")


def test_combiner_weights_are_scale_free() -> None:
    from pref_dispatch.llm.combiner_adapter import LLMCombiner

    class _Scorer:
        def __init__(self, mult):
            self.mult = mult

        def skill_scores(self, driver_obs, phi_ep, phi_step, w):
            return {"a": 3.0 * self.mult, "b": 1.0 * self.mult, "c": 0.0}

    names = ("a", "b", "c")
    small = LLMCombiner(_Scorer(1.0), names, blend_k=2).weights_for({}, PHI_EP, PHI_STEP)
    big = LLMCombiner(_Scorer(1000.0), names, blend_k=2).weights_for({}, PHI_EP, PHI_STEP)
    assert set(small) == {"a", "b"}, small
    assert all(abs(small[k] - big[k]) < 1e-9 for k in small), (small, big)
    hot = LLMCombiner(_Scorer(1.0), names, blend_k=1).weights_for({}, PHI_EP, PHI_STEP)
    assert hot == {"a": 1.0}, hot
    print(f"[d] combiner weights OK: k=2 -> {small} at both x1 and x1000; "
          f"k=1 -> {hot}")


if __name__ == "__main__":
    test_blend_k1_is_raw_one_hot()
    test_normalization_makes_weights_real()
    test_any_skill_floor_drops_the_order()
    test_combiner_weights_are_scale_free()
    test_compute_bids_end_to_end()
    print("\nall blend_k checks passed")
