"""Fairness budget from historical income (proposal 4.5).

Before matching, each driver's per-driver softmax scores are multiplied by a
scalar budget ``beta_d``. Low-income drivers get a larger budget, so they win
more matches -- an auction-style mechanism that pulls driver incomes together.

M1 uses **historical income only** (no future-earnings estimate), by design:
including future potential would require a value function and drag us back
toward RL (proposal risk-five). ``fairness_strength >= 0`` (the preference's
fairness weight) scales how aggressively budgets diverge, letting us sweep the
efficiency-fairness frontier.

**Strength is UNBOUNDED above** (2026-08-10). It used to be clipped to ``[0, 1]``,
which silently flattened every Phase-3 training draw above 1.0 onto exactly 1.0 --
the loop believed it was best-responding across a spread of strengths while half
the batch was the same point. The mechanism itself is well-behaved at any
strength (``exp`` of a z-score is finite and positive for any finite input), and
the limit is the intended one: as strength grows the match order becomes a pure
income ranking. Reported experiments still sweep ``[0, 1]`` -- that is an
experimental choice, not a mechanism constraint.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


class FairnessBudget:
    """Maps historical cumulative income -> per-driver multiplicative budget."""

    def __init__(self, strength: float = 0.0, eps: float = 1e-6):
        # strength=0 => every budget is 1.0 (fairness mechanism off). Negative is
        # clamped away (it would BOOST the already-rich, which no caller means);
        # above, nothing is clipped -- see the module docstring.
        self.strength = max(0.0, float(strength))
        self.eps = eps

    def budgets(self, income: Dict[int, float]) -> Dict[int, float]:
        """Return ``{driver_id: beta_d}`` from cumulative income so far.

        Budgets are centred on 1.0 and inversely tied to income rank: a driver
        earning below the fleet mean gets ``beta > 1``, above the mean ``beta <
        1``, with the spread controlled by ``strength``. With ``strength == 0``
        all budgets are exactly 1.0 (no fairness pressure).
        """
        if self.strength <= 0.0 or not income:
            return {d: 1.0 for d in income}

        ids = list(income.keys())
        vals = np.array([income[d] for d in ids], dtype=float)
        mean = vals.mean()
        scale = vals.std() + self.eps
        # z > 0 for rich drivers; we want them damped, poor drivers boosted.
        z = (vals - mean) / scale
        # exp(-strength * z): poor (z<0) -> >1, rich (z>0) -> <1. Bounded, smooth.
        beta = np.exp(-self.strength * z)
        return {d: float(b) for d, b in zip(ids, beta)}
