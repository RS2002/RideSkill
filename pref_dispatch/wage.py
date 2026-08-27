"""Driver wage (take-home fare), decoupled from the shaping reward.

Feature 2 (driver-income fairness) equalizes what drivers *earn*, not the
shaping reward. The shaping reward (:class:`ride_gym.rewards.DefaultRewardFunction`)
nets in penalties -- idle, empty-move, detour, end-to-end service time -- so its
per-driver cumulative sum ("income") is NOT money; a driver punished for idling
looks poorer than one who simply took fewer fares. Fairness on that quantity
would equalize *penalties*, which is not what "make driver wages fair" means.

The **wage** is the pure fare the platform pays a driver for the trips it is
ASSIGNED: the same ``solo_time x party`` term the reward's revenue bonus and
:class:`pref_dispatch.metrics.EpisodeMetrics` (fleet ``revenue``) already use,
but kept per-driver and stripped of every penalty. Booked at assignment time
(when the trip is committed to the driver), matching where the revenue bonus is
credited in the reward.

This is the single source of truth for "what a driver earns" -- both rollout
paths (online :func:`pref_dispatch.evaluate.rollout` and the benchmark
:class:`benchmark.recorder.EpisodeRecorder`) accumulate wage through this one
function, so the fairness mechanism and the ``wage_gini`` KPI always agree on
the definition.
"""

from __future__ import annotations

from typing import Dict


def driver_wage_from_event(event: Dict) -> float:
    """Fare a driver earns this step: sum over newly assigned orders of
    ``solo_time x party`` (direct pickup->dropoff time times passenger count).

    Pure money -- no idle / empty-move / detour / service-time penalties. Uses
    the same per-order event fields the env populates (see
    :meth:`ride_gym.rewards.RewardFunction.__call__`); returns 0.0 for a step
    with no new assignments (e.g. a pure idle step), where the shaping reward
    would instead be negative.
    """
    solo = event.get("assigned_solo_times", {})
    party = event.get("assigned_party_sizes", {})
    return sum(
        solo.get(oid, 0.0) * party.get(oid, 1)
        for oid in event.get("assigned_orders", [])
    )
