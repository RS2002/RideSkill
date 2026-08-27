"""Render the platform's CURRENT reward function into a prompt-ready spec.

The v1 pure-Phase-1 arm handed the LLM a *hand-written* string constant
(``prompts.pure_phase1.FIXED_REWARD_SPEC``) that mirrors the reward coefficients
by hand -- which silently desyncs if the benchmark coefficients change (they do:
``benchmark/config.py`` sets ``revenue_coef=0.01 / service_time_coef=0.04 /
detour_coef=0.08``, NOT the ``DefaultRewardFunction`` ABC defaults). Here we read
the coefficients LIVE off the actual reward instance the env will grade with, so
the combiner is composed FOR the real reward it is scored against.

Three interchangeable forms of "platform preference" collapse to one text block:

* a concrete ``RewardFunction`` instance  -> :func:`describe_reward` (the headline
  path: real coefficients + the reward's own source);
* a natural-language brief                -> passed through verbatim;
* a weight dict (revenue/service/...)     -> rendered as a weight statement.

:func:`reward_spec` is the single entry point the evolution/report code calls.
"""

from __future__ import annotations

import inspect
from typing import Mapping, Optional, Union

from ride_gym.rewards import DefaultRewardFunction, RewardFunction

RewardLike = Union[RewardFunction, str, Mapping[str, float]]


def describe_reward(reward_fn: RewardFunction) -> str:
    """A faithful, LLM-readable spec of ``reward_fn`` with LIVE coefficients.

    Reads the coefficients off the instance (so the numbers are exactly what the
    env grades with), states the per-driver per-step reward term by term, adds a
    plain-English "what it wants" gloss, and appends the reward's own source for
    ground truth. Falls back gracefully for a non-default ``RewardFunction`` (no
    known coefficients): it still embeds the source so the model can read it.
    """
    src = _safe_getsource(reward_fn)

    if not isinstance(reward_fn, DefaultRewardFunction):
        # Unknown reward: we cannot enumerate coefficients, but the source is the
        # ground truth -- hand it over and ask the model to read it.
        return (
            "The platform grades every driver with this FIXED per-driver, per-step "
            "reward (you do NOT change it). Read its source and infer what it "
            "rewards and penalises, then compose the frozen skills to MAXIMISE its "
            "realised fleet mean:\n\n```python\n" + src + "\n```"
        )

    a = reward_fn.assignment_bonus
    rev = reward_fn.revenue_coef
    svc = reward_fn.service_time_coef
    det = reward_fn.detour_coef
    empty = reward_fn.empty_move_penalty
    idle = reward_fn.idle_penalty

    # Only surface the empty/idle line when they are actually active, so the model
    # is not told to worry about a penalty that is zero in this configuration.
    tail = (
        f"  - {empty:.4g} * (1 if the car MOVES while empty)          [empty-move penalty]\n"
        f"  - {idle:.4g} * (1 if the car IDLES in place)              [idle penalty]\n"
        if (empty or idle)
        else "  (empty-move and idle penalties are zero in this configuration.)\n"
    )

    return (
        "The platform grades every driver with this FIXED per-driver, per-step "
        "reward (you do NOT design it -- it is the SAME reward the RL baselines are "
        "trained on). For a driver at one step it sums:\n\n"
        f"  + {a:.4g} * (number of orders newly ASSIGNED to this driver this step)"
        "     [throughput]\n"
        f"  + {rev:.4g} * (per newly assigned order) its solo (direct pickup->dropoff)\n"
        f"            trip time in minutes * its passenger count       [revenue / fare]\n"
        f"  - {svc:.4g} * (per newly assigned order) its predicted END-TO-END service\n"
        f"            time (request -> planned drop-off)                [service penalty]\n"
        f"  - {det:.4g} * signed re-routing impact on this driver's already-committed\n"
        f"            onboard orders (later deliveries cost, earlier credit)[detour penalty]\n"
        f"{tail}"
        "\nRead what this reward WANTS from a dispatcher, in plain terms:\n"
        f"  - It pays first and foremost for ASSIGNING orders (the +{a:.4g} bonus "
        "dominates): serving demand beats idling. Throughput matters most.\n"
        f"  - Among orders it mildly prefers longer / served-minute-rich fares (the "
        f"+{rev:.4g} revenue term) but PENALISES orders whose end-to-end fulfilment "
        f"drags (the -{svc:.4g} term) -- so it dislikes long pickups and heavy "
        "pooling delay.\n"
        f"  - It protects already-onboard riders: detouring committed deliveries is "
        f"penalised (-{det:.4g}), so reckless pooling is discouraged.\n"
        "Note there is NO completion bonus: credit lands on the ASSIGNMENT step, not "
        "the drop-off. The realised episode objective is the FLEET MEAN of this "
        "per-driver reward accumulated over the hour (``income_mean``). Higher is "
        "better -- compose the frozen skills to maximise it.\n\n"
        "For ground truth, this is the reward's own source (coefficients above are "
        "read live off the instance the env grades with):\n```python\n" + src + "\n```"
    )


def _render_weights(weights: Mapping[str, float]) -> str:
    """Render an explicit weight vector as a preference statement."""
    items = ", ".join(f'{k}={float(v):.4g}' for k, v in weights.items())
    return (
        "The platform preference is given as an explicit weight vector: "
        f"{items}. Higher weight = the platform values that objective more. "
        "Compose the frozen skills so the fleet's behaviour reflects these weights."
    )


def reward_spec(preference: Optional[RewardLike]) -> Optional[str]:
    """Normalise any supported preference FORM into one prompt text block.

    * ``RewardFunction`` instance -> :func:`describe_reward` (live coefficients).
    * ``str``                     -> returned verbatim (a natural-language brief).
    * mapping                     -> a weight-vector statement.
    * ``None``                    -> ``None`` (caller keeps the legacy behaviour).
    """
    if preference is None:
        return None
    if isinstance(preference, RewardFunction):
        return describe_reward(preference)
    if isinstance(preference, str):
        return preference.strip() or None
    if isinstance(preference, Mapping):
        return _render_weights(preference)
    raise TypeError(f"unsupported preference form: {type(preference)!r}")


def _safe_getsource(obj: object) -> str:
    try:
        return inspect.getsource(type(obj)).rstrip()
    except (OSError, TypeError):  # pragma: no cover -- source not available
        return f"# source unavailable for {type(obj).__name__}"


# --------------------------------------------------------------------------- #
# The event dict an LLM-authored reward may read (§Phase-2 reward authoring).   #
# --------------------------------------------------------------------------- #
# The 11 keys of ride_gym.env.RidePoolEnv._new_event -- the complete, guaranteed
# read surface a reward(event) body may touch. Kept here (not imported) so the
# prompt text is stable and self-documenting; a mismatch is caught by
# sandbox.validate_reward, which runs the authored reward on _synthetic_events().
EVENT_KEYS_SPEC = """\
Each step the env hands your reward ONE driver's `event` dict. These 14 keys are
ALWAYS present (values may be empty); read only these:

  - event['assigned_orders']        : list[int] -- order ids newly ASSIGNED to this
                                        driver this step. len(...) is the throughput
                                        signal. Empty on a step with no new assignment.
  - event['assigned_party_sizes']   : dict[order_id -> int] -- passenger count of each
                                        newly assigned order.
  - event['assigned_dispatch_wait'] : dict[order_id -> float] -- minutes the user spent
                                        on the platform BEFORE this dispatch
                                        (now - request_time). A pure waiting cost;
                                        the passenger has not been picked up yet.
  - event['assigned_pickup_times']  : dict[order_id -> float] -- minutes for the driver
                                        to travel from its CURRENT position to the
                                        order's pickup point (predicted, re-optimised).
  - event['assigned_solo_times']    : dict[order_id -> float] -- the pure in-vehicle
                                        ride time (min) from the PICKUP point to the
                                        DROPOFF point on the direct shortest path.
  - event['assigned_service_times'] : dict[order_id -> float] -- predicted END-TO-END
                                        service time (request -> planned drop-off):
                                        dispatch_wait + pickup_time + in-vehicle ride.
                                        Always >= solo_time.
  - event['assigned_detour_times']  : dict[order_id -> float] -- per order, the extra
                                        END-TO-END delivery time from pooling: the
                                        passenger's predicted drop-off time on the
                                        pooled plan MINUS what it would be if the empty
                                        car served only this order directly (direct
                                        driver->pickup travel + solo ride). This includes
                                        BOTH the driver's detour to reach this pickup (behind
                                        other orders) AND the in-vehicle pooling detour.
                                        Positive if pooling made this passenger's trip longer.
  - event['completed_orders']       : list[int] -- order ids dropped off this step.
  - event['picked_up_orders']       : list[int] -- order ids picked up this step.
  - event['distance_moved']         : float -- coordinate units travelled this step.
  - event['time_moved']             : float -- minutes spent moving this step.
  - event['is_empty_move']          : bool -- moved with zero onboard passengers.
  - event['is_idle_wait']           : bool -- stayed put with no tasks.
  - event['extra_detour_time']      : float -- SIGNED minutes of re-routing impact on
                                        already-committed en-route orders (sum of new
                                        minus old predicted drop-off). Positive = new
                                        orders DELAYED existing deliveries; negative =
                                        re-optimisation sped them up; 0 if none.

Only order ids in `assigned_orders` are keys of the `assigned_*` sub-dicts.
Use `.get(oid, default)` when indexing them. There is deliberately no completion
bonus available beyond what you compute from these keys; credit is normally taken
on the ASSIGNMENT step, not the drop-off (see completed_orders if you disagree).

A passenger's full wait is `dispatch_wait + pickup_time`; its ride time is
`solo_time` when unpooled and `solo_time + detour_time` when pooled. The three
wait/ride components (`dispatch_wait`, `pickup_time`, `solo_time`) are SEPARATE
terms -- price each on its own, and optionally weight any of them by
`assigned_party_sizes[oid]` (a group waiting together is more costly than a solo
passenger waiting)."""


def event_spec() -> str:
    """The prompt block describing the per-step ``event`` dict a reward may read.

    Handed to the reward-authoring LLM so it writes ``reward(event) -> float`` over
    exactly the keys the env populates (matched by
    :func:`pref_dispatch.llm.sandbox._synthetic_events`).
    """
    return EVENT_KEYS_SPEC

