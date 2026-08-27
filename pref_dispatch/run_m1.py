"""M1 driver: verify the closed loop runs and the frontier *moves*.

Run:  python -m pref_dispatch.run_m1

It exercises three things the milestone must demonstrate (proposal section 7):

1. **Closed loop is stable** -- every config completes a full episode with
   finite metrics (no crash, no NaN/inf).
2. **Preference changes move the efficiency frontier** -- sweeping the
   revenue<->service preference under the heuristic combiner shifts revenue vs
   mean service time monotonically-ish (revenue up as revenue-weight up).
3. **Fairness budget moves the fairness axis** -- raising the fairness strength
   lowers income Gini (typically trading off some revenue).

It prints a small table for each and asserts the qualitative directions so the
script doubles as a smoke test.

The environment is the REAL Manhattan scenario (real OSM road network + real
FHVHV demand windows), not a synthetic field -- see :mod:`pref_dispatch.nyc_env`.
"""

from __future__ import annotations

from typing import List

from ride_gym import RidePoolEnv

from pref_dispatch.combiner import make_combiner
from pref_dispatch.evaluate import DispatchController, rollout
from pref_dispatch.nyc_env import TIME_OF_DAY, make_nyc_env
from pref_dispatch.preference import Preference


# Real-data env builder: the whole ``pref_dispatch`` stack is network-agnostic,
# so the toy Euclidean/HotspotOrderGenerator field is gone -- ``make_env`` now
# always returns a real-Manhattan env for the requested time-of-day regime.
make_env = make_nyc_env


def _make_env(seed: int = 0) -> RidePoolEnv:
    """Default M1 env: the real shoulder-hour Manhattan window."""
    return make_nyc_env(seed=seed, regime="shoulder")


def _row(label: str, m: dict) -> str:
    return (
        f"{label:>26} | rev {m['revenue']:8.1f} | svc_rate {m['service_rate']:.2f} "
        f"| mean_svc {m['mean_service_time']:6.2f} | gini {m['income_gini']:.3f} "
        f"| inc_min {m['income_min']:7.2f}"
    )


def sweep_efficiency(seed: int = 0) -> List[dict]:
    print("\n=== (2) Efficiency frontier: sweep revenue<->service (fairness=0) ===")
    print(f"{'preference':>26} | metrics")
    combiner = make_combiner("heuristic")
    results = []
    for rev_w in (0.0, 0.25, 0.5, 0.75, 1.0):
        pref = Preference({"revenue": rev_w, "service": 1.0 - rev_w, "fairness": 0.0})
        env = _make_env(seed)
        ctrl = DispatchController(combiner)
        m = rollout(env, ctrl, pref, seed=seed)
        results.append({"rev_w": rev_w, **m})
        print(_row(repr(pref), m))
    return results


def sweep_fairness(seed: int = 0) -> List[dict]:
    print("\n=== (3) Fairness axis: sweep fairness strength (rev=svc=0.5) ===")
    print(f"{'preference':>26} | metrics")
    combiner = make_combiner("heuristic")
    results = []
    for fair in (0.0, 0.5, 1.0):
        pref = Preference({"revenue": 0.5, "service": 0.5, "fairness": fair})
        env = _make_env(seed)
        ctrl = DispatchController(combiner)
        m = rollout(env, ctrl, pref, seed=seed)
        results.append({"fair": fair, **m})
        print(_row(repr(pref), m))
    return results


def sweep_single_skills(seed: int = 0) -> None:
    print("\n=== (1) Basis extremes: single-skill ablation combiners ===")
    print(f"{'combiner':>26} | metrics")
    pref = Preference({"revenue": 0.5, "service": 0.5, "fairness": 0.0})
    for name in ("revenue", "service", "enroute"):
        env = _make_env(seed)
        ctrl = DispatchController(make_combiner("single", skill_name=name))
        m = rollout(env, ctrl, pref, seed=seed)
        print(_row(f"single:{name}", m))


def _finite(m: dict) -> bool:
    import math

    return all(
        isinstance(v, (int, float)) and math.isfinite(v) for v in m.values()
    )


def main() -> None:
    sweep_single_skills()
    eff = sweep_efficiency()
    fair = sweep_fairness()

    # --- Assertions: the milestone's pass/fail conditions. ---
    assert all(_finite(m) for m in eff), "non-finite metric in efficiency sweep"
    assert all(_finite(m) for m in fair), "non-finite metric in fairness sweep"

    # (2) revenue-weight up => revenue proxy not lower at the extreme than at 0.
    assert eff[-1]["revenue"] >= eff[0]["revenue"], (
        "raising revenue preference did not increase revenue "
        f"({eff[-1]['revenue']:.1f} < {eff[0]['revenue']:.1f})"
    )

    # (3) fairness strength up => income Gini not higher at the extreme.
    assert fair[-1]["income_gini"] <= fair[0]["income_gini"] + 1e-6, (
        "raising fairness strength did not reduce income Gini "
        f"({fair[-1]['income_gini']:.3f} > {fair[0]['income_gini']:.3f})"
    )

    print("\nM1 OK: closed loop stable; efficiency + fairness frontiers move.")


if __name__ == "__main__":
    main()
