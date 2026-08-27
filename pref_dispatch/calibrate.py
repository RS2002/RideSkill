"""Task-(a): calibrate the env to a regime with a genuine preference trade-off.

The M1 generalization run revealed that in the default env one policy
(``single:revenue``) nearly Pareto-dominates, so a fixed policy is optimal and
there is nothing for preference-conditioning to exploit. Before spending an LLM
on M2/M3 we must find an env regime where the *best policy genuinely depends on
the preference* (proposal 5.2 validity requirement).

This sweep scores each candidate regime by two tension signals, both computed
from the existing generalization machinery (no method change):

* **pick_diversity** -- number of DISTINCT specialists chosen across a set of
  held-out preferences (>1 means the optimal policy changes with w).
* **fixed_gap** -- mean ``specialist - fixed`` over held-out preferences (how
  much a single preference-ignoring policy loses vs the per-w best). Large =>
  real trade-off that only preference-reading can capture.

A good regime has BOTH high. We rank regimes by ``fixed_gap`` subject to
``pick_diversity >= 2`` and print the table.

Run:  python -m pref_dispatch.calibrate
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np

from pref_dispatch.generalize import (
    ADAPTIVE_NAME,
    candidate_combiners,
    evaluate_matrix,
    scalarize,
    _norm_ranges,
)
from pref_dispatch.nyc_env import TIME_OF_DAY, make_nyc_env
from pref_dispatch.preference import sample_preferences


# Candidate regimes are REAL time-of-day windows on the held-out Manhattan
# scenario (see pref_dispatch.nyc_env.TIME_OF_DAY). The lever that varies the
# preference tension is real demand volume across the day -- off-peak has fleet
# slack (little contention -> weak tension), peak saturates the fleet (scarcity
# -> the objectives genuinely disagree about who to serve). We score each hour
# for whether the best frozen skill actually changes with the preference.
#
# ``order_limit`` caps each replay so the tension *detector* runs in minutes:
# it only needs the relative skill ranking, and the first N orders of a real
# window preserve that ranking while bounding per-rollout cost. The final
# experiments (run_m1 / generalize) run the FULL window, not this cap.
CALIB_ORDER_LIMIT = 3000
# Fleet size for the tension sweep. Overridable via ``CALIB_DRIVERS`` so the
# calibration operating point matches whatever fleet the frozen skills/combiner
# will later be evolved and evaluated at (they must share one operating point --
# see memory skill-eval-operating-point). Default 800 = the realistic-density
# point (peak service_rate ~0.6-0.7), replacing the earlier 150-car scarcity.
CALIB_DRIVERS = int(os.environ.get("CALIB_DRIVERS", "800"))
REGIMES: List[Dict] = [
    {
        "name": regime,
        "regime": regime,
        "order_limit": CALIB_ORDER_LIMIT,
        "num_drivers": CALIB_DRIVERS,
    }
    for regime in TIME_OF_DAY
]


def score_regime(cfg: Dict, seed: int = 0, n_test: int = 4) -> Dict:
    """Measure ENV-INTRINSIC preference tension, independent of any combiner.

    Tension is a property of the env + the frozen basis skills only, so we
    restrict the pool to single skills and exclude the (mock) adaptive combiner
    entirely -- otherwise a weak mock drags the gap toward zero and we would be
    measuring combiner quality, not env tension.

    * pick_diversity : # distinct single skills that are best across held-out w.
    * fixed_gap      : mean over w of [best single skill for THIS w] minus [the
      single skill that is best ON AVERAGE across all w]. This is exactly the
      specialist-vs-fixed gap the real experiment reports, but with a genuine
      fixed baseline (one skill committed for all w), so it is not inflated by
      per-w cherry-picking of the fixed policy.
    """
    params = {k: v for k, v in cfg.items() if k != "name"}
    env_factory = lambda s: make_nyc_env(seed=s, **params)  # noqa: E731

    W = sample_preferences(n_test, seed=seed + 2)
    cands = {k: v for k, v in candidate_combiners().items() if k != ADAPTIVE_NAME}
    mat = evaluate_matrix(env_factory, cands, W, seed=seed)
    ranges = _norm_ranges(mat)

    per_w = {pi: {c: scalarize(mat[(c, pi)], W[pi], ranges) for c in cands}
             for pi in range(len(W))}

    # Best single skill on average -> the honest fixed baseline for this regime.
    avg = {c: float(np.mean([per_w[pi][c] for pi in per_w])) for c in cands}
    fixed_skill = max(avg, key=avg.get)

    picks: List[str] = []
    gaps: List[float] = []
    for pi in per_w:
        spec = max(per_w[pi], key=per_w[pi].get)
        picks.append(spec)
        gaps.append(per_w[pi][spec] - per_w[pi][fixed_skill])

    return {
        "name": cfg["name"],
        "pick_diversity": len(set(picks)),
        "picks": picks,
        "fixed_skill": fixed_skill,
        "fixed_gap": float(np.mean(gaps)),
    }


def main() -> None:
    # ``CALIB_FULL=1`` drops the order cap for the definitive full-demand run
    # (each regime replays its ENTIRE real hour window). Default stays capped so
    # the sweep is a fast tension *detector*.
    full = os.environ.get("CALIB_FULL") == "1"
    regimes = [
        {**cfg, "order_limit": None} if full else cfg for cfg in REGIMES
    ]
    tag = "FULL demand (entire hour windows)" if full else \
        f"capped at {CALIB_ORDER_LIMIT} orders"
    print(f"Calibrating env regimes for a genuine preference trade-off "
          f"[{tag}]...\n", flush=True)

    # Stream each regime to disk the instant it finishes, so an interrupted run
    # never loses completed regimes (main() used to print only at the very end).
    out_path = os.path.join(
        "pref_dispatch", "calib_full.jsonl" if full else "calib_capped.jsonl"
    )
    rows: List[Dict] = []
    print(f"{'regime':>18} | pick_div | fixed_gap | distinct picks", flush=True)
    with open(out_path, "w") as fh:
        for cfg in regimes:
            r = score_regime(cfg)
            rows.append(r)
            distinct = ",".join(sorted({p.split(":")[-1] for p in r["picks"]}))
            print(
                f"{r['name']:>18} | {r['pick_diversity']:^8d} | {r['fixed_gap']:9.3f} "
                f"| {distinct}",
                flush=True,
            )
            fh.write(json.dumps(r) + "\n")
            fh.flush()
    print(f"\n(results streamed to {out_path})", flush=True)

    good = [r for r in rows if r["pick_diversity"] >= 2 and r["fixed_gap"] > 0.05]
    print()
    if good:
        best = max(good, key=lambda r: r["fixed_gap"])
        print(
            f"Recommended regime: '{best['name']}' "
            f"(pick_diversity={best['pick_diversity']}, "
            f"fixed_gap={best['fixed_gap']:.3f})."
        )
        print("Adopt its params as the default env for M2/M3.")
    else:
        print(
            "No regime yet shows a strong trade-off (pick_diversity>=2 AND "
            "fixed_gap>0.05). Widen the sweep (more extreme map/speed, sharper "
            "hotspot, or a competing objective the skills disagree on)."
        )


if __name__ == "__main__":
    main()
