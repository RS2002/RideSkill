"""Preference-generalization harness (proposal 5.2), M1-stage scaffolding.

This builds the *measurement* frame for the headline experiment BEFORE any LLM
is involved. The crucial observation is that the current mock combiner already
reads the preference ``w`` as a formal argument, so it is a zero-retrain
function ``theta(w)`` -- exactly the role the LLM-evolved combiner will occupy in
M3. Hence the three comparison curves the paper needs can be computed now:

* **specialist(w)**  -- best policy chosen *per w* from a candidate pool. A
  stand-in for "retrain a specialist per preference" (the gold-standard upper
  bound). Later replaced by per-w retrained MARL / evolved scorers.
* **adaptive(w)**    -- ONE preference-reading function evaluated at each w (the
  heuristic combiner now; the LLM combiner in M3). This is *our method*.
* **fixed(w)**       -- the single best preference-*ignoring* policy chosen on
  ``W_train`` and applied to all of ``W_test`` (the ablation lower bound).

Reported: the generalization gap ``specialist - adaptive`` on held-out
preferences (small = good), and ``specialist - fixed`` (should be larger,
showing preference-conditioning helps). M3 succeeds if the LLM adaptive shrinks
its gap below the heuristic's.

Run:  python -m pref_dispatch.generalize
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np

from pref_dispatch.combiner import Combiner, make_combiner
from pref_dispatch.evaluate import DispatchController, rollout
from pref_dispatch.preference import Preference, sample_preferences

# Metric orientation: +1 = higher is better, -1 = lower is better. Only the
# terms that enter the scalarised objective need to appear here.
METRIC_SENSE = {"revenue": +1, "mean_service_time": -1, "income_gini": -1}


def candidate_combiners() -> Dict[str, Combiner]:
    """The discrete policy pool the specialist selects from per preference.

    In M1 this is: the three single-skill extremes (preference-ignoring), plus
    the preference-reading heuristic. M2/M3 grow this pool with LLM-evolved
    skills and swap the heuristic for the LLM combiner.
    """
    return {
        "single:revenue": make_combiner("single", skill_name="revenue"),
        "single:service": make_combiner("single", skill_name="service"),
        "single:enroute": make_combiner("single", skill_name="enroute"),
        "adaptive:heuristic": make_combiner("heuristic"),
    }


ADAPTIVE_NAME = "adaptive:heuristic"  # the (currently mock) preference-reader.


def evaluate_matrix(
    env_factory: Callable,
    candidates: Dict[str, Combiner],
    prefs: List[Preference],
    seed: int = 0,
) -> Dict[Tuple[str, int], Dict[str, float]]:
    """Run every (candidate, preference) once; return metrics keyed by both.

    ``env_factory(seed)`` must return a fresh env so all cells share the same
    demand instance (fair comparison). ``rollout`` reseeds on reset.
    """
    results: Dict[Tuple[str, int], Dict[str, float]] = {}
    for name, comb in candidates.items():
        for pi, pref in enumerate(prefs):
            env = env_factory(seed)
            ctrl = DispatchController(comb)
            results[(name, pi)] = rollout(env, ctrl, pref, seed=seed)
    return results


def _norm_ranges(
    matrix: Dict[Tuple[str, int], Dict[str, float]]
) -> Dict[str, Tuple[float, float]]:
    """Min/max of each scalarised metric across the whole matrix (for scaling)."""
    ranges: Dict[str, Tuple[float, float]] = {}
    for m in METRIC_SENSE:
        vals = [cell[m] for cell in matrix.values()]
        ranges[m] = (min(vals), max(vals))
    return ranges


def scalarize(
    metrics: Dict[str, float],
    pref: Preference,
    ranges: Dict[str, Tuple[float, float]],
) -> float:
    """Preference-weighted scalar objective in [0, 1]-ish, higher = better.

    Each metric is min-max normalised across the matrix and oriented so higher
    is better, then combined: efficiency terms by (revenue, service) weights and
    the fairness term by the fairness strength.
    """
    def n(metric: str) -> float:
        lo, hi = ranges[metric]
        span = hi - lo
        x = (metrics[metric] - lo) / span if span > 1e-12 else 0.5
        return x if METRIC_SENSE[metric] > 0 else 1.0 - x

    eff = pref["revenue"] * n("revenue") + pref["service"] * n("mean_service_time")
    fair = pref["fairness"] * n("income_gini")
    return eff + fair


def run(env_factory: Callable, seed: int = 0, n_train: int = 6, n_test: int = 6):
    W_train = sample_preferences(n_train, seed=seed + 1)
    W_test = sample_preferences(n_test, seed=seed + 2)
    cands = candidate_combiners()

    train_mat = evaluate_matrix(env_factory, cands, W_train, seed=seed)
    test_mat = evaluate_matrix(env_factory, cands, W_test, seed=seed)

    # Normalisation ranges from the union so train/test scores are comparable.
    union = dict(train_mat)
    union.update({(f"test:{k[0]}", k[1]): v for k, v in test_mat.items()})
    ranges = _norm_ranges(union)

    # Fixed lower bound: the single preference-IGNORING candidate that is best on
    # average over W_train (the heuristic reads w, so exclude it here).
    fixed_pool = [c for c in cands if c != ADAPTIVE_NAME]
    fixed_best, fixed_best_score = None, -1e18
    for name in fixed_pool:
        avg = np.mean(
            [scalarize(train_mat[(name, pi)], W_train[pi], ranges)
             for pi in range(len(W_train))]
        )
        if avg > fixed_best_score:
            fixed_best, fixed_best_score = name, avg

    print(f"\nFixed (preference-ignoring) baseline chosen on W_train: {fixed_best}")
    print(f"{'w (held-out)':>26} | specialist  adaptive  fixed  | gap_ad  gap_fx  pick")
    gaps_ad, gaps_fx = [], []
    for pi, w in enumerate(W_test):
        scored = {name: scalarize(test_mat[(name, pi)], w, ranges) for name in cands}
        spec_name = max(scored, key=scored.get)
        spec = scored[spec_name]
        adapt = scored[ADAPTIVE_NAME]
        fixed = scored[fixed_best]
        gaps_ad.append(spec - adapt)
        gaps_fx.append(spec - fixed)
        print(
            f"{repr(w):>26} | {spec:9.3f} {adapt:9.3f} {fixed:7.3f} "
            f"| {spec-adapt:6.3f} {spec-fixed:6.3f}  {spec_name.split(':')[-1]}"
        )

    mean_ad, mean_fx = float(np.mean(gaps_ad)), float(np.mean(gaps_fx))
    print(f"\nMean held-out gap  adaptive={mean_ad:.3f}  fixed={mean_fx:.3f}")
    print(
        "Interpretation: adaptive gap < fixed gap => reading the preference "
        "helps generalise. M3 target: LLM adaptive gap < heuristic's."
    )
    return {"gap_adaptive": mean_ad, "gap_fixed": mean_fx}


def _specialist_varies(env_factory, seed=0, n=6) -> bool:
    """Does the per-w specialist pick actually change with w?

    If one candidate wins for *every* preference, the environment has no real
    trade-off and the generalization experiment has no teeth here -- a fixed
    policy is optimal by construction. This is the property the experiment
    design must guarantee (proposal 5.2), so we surface it explicitly rather
    than let a degenerate env masquerade as a passing result.
    """
    W = sample_preferences(n, seed=seed + 2)
    cands = candidate_combiners()
    mat = evaluate_matrix(env_factory, cands, W, seed=seed)
    ranges = _norm_ranges(mat)
    picks = {
        max(
            {c: scalarize(mat[(c, pi)], W[pi], ranges) for c in cands}.items(),
            key=lambda kv: kv[1],
        )[0]
        for pi in range(len(W))
    }
    return len(picks) > 1


def main() -> None:
    from pref_dispatch.run_m1 import _make_env

    res = run(_make_env)

    # Harness sanity: gaps must be finite and non-negative (specialist is the
    # per-w max, so it can never be beaten by any single candidate).
    import math

    assert all(math.isfinite(v) for v in res.values()), "non-finite gap"
    assert res["gap_adaptive"] >= -1e-9 and res["gap_fixed"] >= -1e-9, (
        "specialist should upper-bound every candidate by construction"
    )

    # Experiment-validity check (NOT a research claim): the env must present a
    # genuine trade-off, else there is nothing for preference-conditioning to
    # exploit. This is what the M1 result above revealed is currently weak.
    varies = _specialist_varies(_make_env)
    print(f"\nSpecialist choice varies with preference: {varies}")
    if not varies:
        print(
            "WARNING: one policy wins for all preferences -> no trade-off in "
            "this env. The generalization experiment needs a demand regime "
            "where the best policy genuinely depends on w (see notes)."
        )
    print("\nGeneralization harness OK (scaffolding validated).")


if __name__ == "__main__":
    main()
