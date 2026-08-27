"""M1 closed-loop skeleton for the preference-adaptive LLM dispatch proposal.

This package is the *milestone-1* validation harness described in
``proposal.md`` section 7. Everything here is intentionally **mock / handwritten**
(no LLM yet). The single goal of M1 is to confirm that the closed loop

    global_stats -> upper combiner -> frozen lower skills
        -> per-driver softmax(+no-op) x fairness budget
        -> per-step one-to-one bipartite matching -> RidePoolEnv.step()

runs without diverging, and that *changing the combiner / fairness strength
moves the outcome on the efficiency-fairness frontier*. LLM proposers replace
the mock combiner (upper) and mock skills (lower) in M2/M3.

Module map (mirrors the proposal's method section):

* :mod:`pref_dispatch.preference` -- platform preference ``w`` (reward weights).
* :mod:`pref_dispatch.global_stats` -- global feature vector ``phi``.
* :mod:`pref_dispatch.skills` -- frozen lower-layer scoring skill basis.
* :mod:`pref_dispatch.combiner` -- mock preference-conditioned upper combiner.
* :mod:`pref_dispatch.budget` -- fairness budget from historical income.
* :mod:`pref_dispatch.matching` -- softmax + budget + greedy 1:1 matching.
* :mod:`pref_dispatch.metrics` -- episode metrics incl. income Gini.
* :mod:`pref_dispatch.evaluate` -- rollout loop: given ``w`` -> metrics.
"""

from pref_dispatch.preference import Preference
from pref_dispatch.evaluate import DispatchController, rollout

__all__ = ["Preference", "DispatchController", "rollout"]
