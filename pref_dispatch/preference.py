"""Platform preference ``w`` -- the weights over reward terms.

A preference is a normalised weight vector over the objective terms the platform
cares about. In M1 we expose three canonical terms that the frozen skills and
the metrics both understand:

* ``revenue``      -- platform revenue / driver earnings (maximise).
* ``service``      -- passenger service time, i.e. pickup + detour (minimise).
* ``fairness``     -- how strongly the platform wants driver income equalised.

The proposal's headline experiment (section 5.2) trains the upper combiner on a
set ``W_train`` of these vectors and evaluates on a held-out ``W_test``. The
simplex sampling helper here is what produces those sets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# Canonical objective terms. Order is fixed so a preference can also be treated
# as a plain 3-vector when convenient.
TERMS: Tuple[str, ...] = ("revenue", "service", "fairness")


@dataclass(frozen=True)
class Preference:
    """A normalised weight vector over :data:`TERMS`.

    ``weights`` maps each term name to a non-negative weight; the efficiency
    terms (``revenue``/``service``) are renormalised to sum to 1 so they define
    a point on the 2-term efficiency simplex, while ``fairness`` is kept as a
    separate strength knob (it scales the budget mechanism, not the scorer, so it
    is not part of the efficiency trade-off itself).

    ``fairness`` is floored at 0 and **not** capped (2026-08-10). It used to be
    clipped to ``[0, 1]`` here AND again in
    :class:`~pref_dispatch.budget.FairnessBudget`; the two clips together meant a
    Phase-3 training draw of 1.7 was silently the same experiment as one of 1.0.
    Reported experiments still sweep ``[0, 1]`` -- an experimental choice, not a
    mechanism constraint.
    """

    weights: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        w = {t: float(self.weights.get(t, 0.0)) for t in TERMS}
        eff = w["revenue"] + w["service"]
        if eff <= 0:
            w["revenue"], w["service"] = 0.5, 0.5
        else:
            w["revenue"] /= eff
            w["service"] /= eff
        # Floor only: a negative strength would BOOST the already-rich, which no
        # caller means. No upper cap -- see the class docstring.
        w["fairness"] = max(0.0, float(w["fairness"]))
        # dataclass is frozen: assign through object.__setattr__.
        object.__setattr__(self, "weights", w)

    def __getitem__(self, term: str) -> float:
        return self.weights[term]

    def get(self, term: str, default: float = 0.0) -> float:
        """Read a weight with a fallback (dict-style read access).

        Evolved/LLM-authored ``skill_scores`` and combiner code naturally reach
        for ``pref.get("revenue", 0.5)``; supporting it keeps that very common
        access from crashing an otherwise-valid candidate. Read-only: the
        preference stays frozen.
        """
        return self.weights.get(term, default)

    def vector(self) -> np.ndarray:
        return np.array([self.weights[t] for t in TERMS], dtype=float)

    def __repr__(self) -> str:
        return (
            "Pref(rev={revenue:.2f}, svc={service:.2f}, "
            "fair={fairness:.2f})".format(**self.weights)
        )


def sample_preferences(
    n: int, seed: int = 0, fairness_levels: Tuple[float, ...] = (0.0, 0.5, 1.0)
) -> List[Preference]:
    """Sample ``n`` preferences: efficiency split on the simplex x fairness grid.

    Used to build ``W_train`` / ``W_test``. The efficiency split (revenue vs
    service) is drawn uniformly on the 1-simplex; the fairness strength is drawn
    from a small fixed grid so both extremes and the middle are represented.
    """
    rng = np.random.default_rng(seed)
    prefs: List[Preference] = []
    for _ in range(n):
        rev = float(rng.uniform(0.0, 1.0))
        fair = float(rng.choice(fairness_levels))
        prefs.append(
            Preference({"revenue": rev, "service": 1.0 - rev, "fairness": fair})
        )
    return prefs
