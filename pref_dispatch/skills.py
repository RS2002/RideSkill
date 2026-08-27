"""Frozen lower-layer scoring skill basis (proposal 4.3; final-version signature).

Each skill is a fixed function

    score(driver_obs, order, phi_ep, phi_step) -> float   # (driver, order) value
    noop_score(driver_obs, phi_ep, phi_step)   -> float   # value of NOT bidding

that returns the **marginal value of inserting ``order`` into the driver's
current route** under one particular objective specialisation. In M1 these are
*handwritten* (the LLM evolves/replaces them in M2). They deliberately span the
single-objective extremes so the upper combiner has a basis that can cover the
preference simplex (proposal risk-one / coverage requirement):

* :class:`RevenueSkill`  -- chase long, high-party fares; tolerate distance.
* :class:`ServiceSkill`  -- minimise pickup + ride time; refuse far/long orders.
* :class:`EnRouteSkill`  -- protect already-onboard orders near their deadline;
  only accept low-detour additions, otherwise prefer the no-op.

Two-layer global context (final-version redesign):

* ``phi_ep`` (:class:`~pref_dispatch.global_stats.EpisodeStats`) -- episode-static.
  It carries the travel-time closure ``phi_ep.dist`` (the road network is fixed
  for the episode, so the skill no longer takes ``dist`` as a separate argument)
  and the leak-free static map ``phi_ep.scale``.
* ``phi_step`` (:class:`~pref_dispatch.global_stats.GlobalStats`) -- live per-step
  state: ``phi_step.mean_solo_time`` (the live scale), ``phi_step.demand_pressure``,
  and the per-region ``kappa`` arrays.

Skills stay **objective-specialist**: they never see the episode objective ``w``
(only the combiner + repositioner do), so the skill library cannot collapse into a
single universal maximiser.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from pref_dispatch.global_stats import EpisodeStats, GlobalStats

Dist = Callable[[tuple, tuple], float]


def _feasible(driver_obs: Dict, order: Dict) -> bool:
    """Capacity feasibility: order party fits in remaining committed budget."""
    s = driver_obs["self"]
    free = s["capacity"] - s["committed_passengers"]
    return order["num_passengers"] <= free


def _pickup_time(driver_obs: Dict, order: Dict, dist: Dist) -> float:
    return dist(driver_obs["self"]["location"], order["origin"])


def _solo_time(order: Dict, dist: Dist) -> float:
    return dist(order["origin"], order["destination"])


def _onboard_slack(driver_obs: Dict) -> float:
    """Smallest remaining eta among the driver's committed orders.

    A small slack means an onboard order is close to its predicted drop-off /
    at risk if the route grows -- the en-route skill uses this to become
    protective. Returns +inf when the driver carries nothing.
    """
    etas = [
        d["eta"]
        for d in driver_obs["self"]["assigned_order_details"]
        if d.get("eta") is not None
    ]
    return min(etas) if etas else float("inf")


def _scale(phi_ep: EpisodeStats, phi_step: GlobalStats) -> float:
    """The live map scale skills threshold in.

    Prefer the live pending-pool scale (``phi_step.mean_solo_time``); fall back to
    the episode-static ``phi_ep.scale`` when the pending pool is empty (so the unit
    is well-defined even at an idle step). Both are in travel-time minutes.
    """
    live = getattr(phi_step, "mean_solo_time", 0.0)
    if live and live > 1e-6:
        return live
    return max(getattr(phi_ep, "scale", 1.0), 1e-6)


class Skill:
    """Base class: a named, frozen scoring function."""

    name = "base"

    def score(
        self, driver_obs: Dict, order: Dict, phi_ep: EpisodeStats, phi_step: GlobalStats
    ) -> float:
        raise NotImplementedError

    def noop_score(
        self, driver_obs: Dict, phi_ep: EpisodeStats, phi_step: GlobalStats
    ) -> float:
        """Baseline value of bidding on nothing. 0.0 unless a skill overrides."""
        return 0.0

    def __repr__(self) -> str:
        return f"Skill({self.name})"


class RevenueSkill(Skill):
    """Maximise platform revenue: long solo trips x party size pay more.

    All quantities are expressed in units of the live map scale so the score is
    O(1) regardless of the map scale -- that is what makes the fixed no-op
    threshold meaningful across environments (scale-adaptive via ``phi_step``).
    Distance to pickup is only lightly penalised, so this skill will happily send
    an empty car far to grab a lucrative fare; under demand pressure it leans in
    further (scarce supply => grab the money).
    """

    name = "revenue"

    def score(self, driver_obs, order, phi_ep, phi_step):
        if not _feasible(driver_obs, order):
            return -1e9
        dist = phi_ep.dist
        scale = _scale(phi_ep, phi_step)
        revenue = (_solo_time(order, dist) / scale) * order["num_passengers"]
        pickup = _pickup_time(driver_obs, order, dist) / scale
        greed = 1.0 + 0.5 * phi_step.demand_pressure
        return greed * revenue - 0.2 * pickup

    def noop_score(self, driver_obs, phi_ep, phi_step):
        # Revenue-hungry: bidding almost always beats idling. Low floor.
        return 0.1


class ServiceSkill(Skill):
    """Minimise passenger service time: prefer near, short orders; else wait."""

    name = "service"

    def score(self, driver_obs, order, phi_ep, phi_step):
        if not _feasible(driver_obs, order):
            return -1e9
        dist = phi_ep.dist
        scale = _scale(phi_ep, phi_step)
        pickup = _pickup_time(driver_obs, order, dist) / scale
        ride = _solo_time(order, dist) / scale
        # Value is highest for a nearby short trip. Centred near 0 so that only
        # better-than-average orders beat the no-op floor below.
        return 1.5 - (1.0 * pickup + 0.5 * ride)

    def noop_score(self, driver_obs, phi_ep, phi_step):
        # Wait unless a genuinely cheap-to-serve order shows up. In the same
        # (scale-free) units as score(): orders worse than ~average lose to this.
        return 0.5


class EnRouteSkill(Skill):
    """Protect committed onboard orders near their deadline.

    If any onboard order has little eta slack, adding stops risks its delivery,
    so the no-op becomes attractive and only tiny-detour additions are accepted.
    An empty / slack-rich driver behaves close to a mild service skill. Slack is
    measured in scale units too, so the deadline sensitivity is map-independent.
    """

    name = "enroute"

    def score(self, driver_obs, order, phi_ep, phi_step):
        if not _feasible(driver_obs, order):
            return -1e9
        dist = phi_ep.dist
        scale = _scale(phi_ep, phi_step)
        pickup = _pickup_time(driver_obs, order, dist) / scale
        ride = _solo_time(order, dist) / scale
        slack = _onboard_slack(driver_obs) / scale
        # Detour aversion grows as slack shrinks (protect the deadline).
        detour_aversion = 1.0 + 3.0 / (1.0 + max(0.0, slack))
        added = pickup + ride
        return 1.5 - detour_aversion * added

    def noop_score(self, driver_obs, phi_ep, phi_step):
        slack = _onboard_slack(driver_obs) / _scale(phi_ep, phi_step)
        if slack == float("inf"):
            # Empty car: behave like a mild service skill (modest wait floor).
            return 0.5
        # Tight slack => strongly prefer not to grow the route this step.
        return 0.5 + 3.0 / (1.0 + max(0.0, slack))


def default_skill_basis() -> List[Skill]:
    """The frozen M1 basis, spanning the single-objective extremes."""
    return [RevenueSkill(), ServiceSkill(), EnRouteSkill()]
