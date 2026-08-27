"""Mock preference-conditioned upper-layer combiner (proposal 4.4; final signature).

The upper layer maps ``(driver_state, phi_ep, phi_step, w)`` -> the scoring
function that driver uses this step, by classifying the driver and picking /
blending the frozen lower skills. In M1 this is a **handwritten** stand-in for
the LLM-evolved combiner; its only job is to make the closed loop exercisable
and to show that *different combiners move the outcome on the frontier*.

Final-version redesign: the combiner is the layer that reads the episode
objective ``w`` (an LLM-authored callable reward function carried on ``phi_ep``;
``None`` = objective-blind). Skills never see ``w`` -- only the combiner and the
repositioner do -- so the skill library stays a set of objective specialists and
the combiner is what specialises the *blend* to the objective. A combiner MAY call
``w`` on a probe event to self-derive the blend (the ``reward_aware_dispatcher_v2``
pattern), or fall back to the platform preference when ``w is None``.

Three mock combiners are provided:

* :class:`SingleSkillCombiner` -- ablation: every driver always uses one fixed
  skill (no classification). Lets us check the basis extremes.
* :class:`EqualBlendCombiner` -- ablation: every skill gets the same weight for
  every driver under every objective. This is the "no choice was made" policy,
  and Phase-2 training subtracts its reward so a fitness of 0 means "worth
  exactly as much as not choosing".
* :class:`HeuristicCombiner` -- the real M1 mock: it (a) classifies each driver
  by state (deadline-pressed / loaded / idle) and (b) weights the resulting
  skill mix by the platform preference. This is the behaviour the LLM combiner
  will later replace.

A combiner returns, per driver, a **weight dict over skill names**. The matcher
blends the named skills' scores by these weights before softmax, so a combiner
can either hard-select (one weight = 1) or soft-blend.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from pref_dispatch.global_stats import EpisodeStats, GlobalStats
from pref_dispatch.preference import Preference

RewardFn = Callable[[Dict], float]


class Combiner:
    """Base: given a driver obs + phi_ep + phi_step + objective w, return weights."""

    def weights_for(
        self,
        driver_obs: Dict,
        phi_ep: EpisodeStats,
        phi_step: GlobalStats,
        w: Optional[RewardFn] = None,
    ) -> Dict[str, float]:
        raise NotImplementedError

    def classify(
        self, driver_obs: Dict, phi_ep: EpisodeStats, phi_step: GlobalStats
    ) -> str:
        """Human-readable driver class (for interpretability / logging)."""
        return "all"


class SingleSkillCombiner(Combiner):
    """Ablation combiner: everyone always uses ``skill_name``."""

    def __init__(self, skill_name: str):
        self.skill_name = skill_name

    def weights_for(self, driver_obs, phi_ep, phi_step, w=None):
        return {self.skill_name: 1.0}

    def classify(self, driver_obs, phi_ep, phi_step):
        return self.skill_name


class EqualBlendCombiner(Combiner):
    """Ablation combiner: every frozen skill gets the SAME weight, always.

    This is the "the upper layer contributed nothing" policy -- the Phase-2
    analogue of switching repositioning OFF in Phase 3. It still dispatches (you
    must), but its weights do not depend on the driver, the step, or the
    objective, so any reward a real combiner earns ABOVE this one is reward that
    came from *choosing*. :mod:`pref_dispatch.llm.combiner_eval` rolls it once per
    cell and subtracts it, which is what makes a Phase-2 fitness of 0.0 mean
    "worth exactly as much as not choosing" instead of "average of whoever else
    happened to be alive this round".

    ``blend_k`` is carried on the instance because
    :class:`~pref_dispatch.evaluate.DispatchController` reads it off the combiner:
    the baseline must be truncated by the same rule as the candidates it is
    compared against, or the two are not the same policy. With equal weights
    :func:`~pref_dispatch.matching._active_skills` breaks the tie by name, so the
    surviving set is a FIXED, driver- and objective-independent slice of the
    library -- which is exactly the property the baseline needs.
    """

    def __init__(self, skill_names: List[str], *, blend_k: int = 1):
        self.skill_names = tuple(skill_names)
        if not self.skill_names:
            raise ValueError("EqualBlendCombiner needs at least one skill name.")
        self.blend_k = int(blend_k)

    def weights_for(self, driver_obs, phi_ep, phi_step, w=None):
        return {name: 1.0 for name in self.skill_names}

    def classify(self, driver_obs, phi_ep, phi_step):
        return "equal_blend"


class HeuristicCombiner(Combiner):
    """Preference-conditioned, driver-class-aware mock combiner.

    Driver classes (LLM will later discover these; here they are handwritten):

    * ``pressed``  -- carries an onboard order with little eta slack -> lean on
      the en-route protection skill regardless of preference.
    * ``loaded``   -- carries orders but with comfortable slack -> blend service
      and en-route.
    * ``idle``     -- empty car -> blend revenue vs service by the platform's
      efficiency preference.

    The preference shifts the blend continuously, which is exactly the
    "adapt to platform preference without retraining" behaviour we want to
    demonstrate moves the frontier. When an objective ``w`` is supplied the LLM
    combiner would self-derive from it; this handwritten mock reads the platform
    ``pref`` it was constructed with (default neutral) as a stand-in.
    """

    def __init__(self, slack_threshold: float = 5.0, pref: Optional[Preference] = None):
        self.slack_threshold = slack_threshold
        self.pref = pref or Preference(
            weights={"revenue": 0.5, "service": 0.5, "fairness": 0.0}
        )

    def classify(self, driver_obs, phi_ep, phi_step):
        details = driver_obs["self"]["assigned_order_details"]
        if not details:
            return "idle"
        etas = [d["eta"] for d in details if d.get("eta") is not None]
        min_slack = min(etas) if etas else float("inf")
        if min_slack <= self.slack_threshold:
            return "pressed"
        return "loaded"

    def weights_for(self, driver_obs, phi_ep, phi_step, w=None):
        cls = self.classify(driver_obs, phi_ep, phi_step)
        rev, svc = self.pref["revenue"], self.pref["service"]

        if cls == "pressed":
            # Protect the deadline: en-route dominates, tiny service seasoning.
            return {"enroute": 0.8, "service": 0.2}
        if cls == "loaded":
            # Comfortable slack: mostly service, but honour revenue appetite.
            return {"service": 0.6 + 0.2 * svc, "enroute": 0.2, "revenue": 0.2 * rev}
        # idle: the platform's efficiency preference decides the blend.
        return {"revenue": rev, "service": svc}


def make_combiner(kind: str, **kwargs) -> Combiner:
    """Factory used by the runner to sweep combiner variants."""
    if kind == "heuristic":
        return HeuristicCombiner(**kwargs)
    if kind == "single":
        return SingleSkillCombiner(kwargs["skill_name"])
    raise ValueError(f"unknown combiner kind: {kind!r}")
