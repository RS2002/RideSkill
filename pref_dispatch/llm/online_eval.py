"""Online rollout for paradigm C: call the LLM once per step, then dispatch.

The paradigm-B :func:`pref_dispatch.evaluate.rollout` assumes a frozen combiner
whose ``weights_for`` is a pure function. Paradigm C's
:class:`~pref_dispatch.llm.paradigm_c.OnlineLLMController` instead needs ONE LLM
query per env step, cached and reused across that step's per-driver matcher calls.
This module provides the thin online loop that inserts the ``begin_step`` query
before the matcher runs, and returns both the episode metrics and the compute
meter -- everything the B/C table needs.

It reuses the exact same matcher / env-step path as B (only the weight *source*
differs), so any effect gap is attributable to the decision policy, not a
different execution mechanism.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from ride_gym import RidePoolEnv

from pref_dispatch.budget import FairnessBudget
from pref_dispatch.evaluate import _make_dist
from pref_dispatch.global_stats import EpisodeStats, GlobalStats
from pref_dispatch.llm.paradigm_c import OnlineLLMController, StepMeter
from pref_dispatch.matching import compute_bids
from pref_dispatch.metrics import EpisodeMetrics
from pref_dispatch.preference import Preference
from pref_dispatch.skills import Skill


def online_rollout(
    env: RidePoolEnv,
    controller: OnlineLLMController,
    skills: Dict[str, Skill],
    pref: Preference,
    *,
    seed: Optional[int] = None,
    temperature: float = 1.0,
    top_k: int = 20,
    max_steps: Optional[int] = None,
) -> Tuple[Dict[str, float], StepMeter]:
    """Run one episode where the LLM assigns skills each step (paradigm C).

    ``max_steps`` caps the number of LLM-driven steps (C is meant to be sampled,
    not run for a full long episode -- §5.5). After the cap the episode ends early
    and metrics are finalised on what ran; the meter reflects only the metered
    steps. Returns ``(metrics, meter)``.
    """
    observations, _info = env.reset(seed=seed)
    dist = _make_dist(env)
    # Episode-static phi_ep (fleet/layout/scale + dist), computed once. Paradigm C
    # is objective-blind at this ablation (w=None); the online controller supplies
    # the per-driver skill choice, not an objective.
    speed = float(getattr(getattr(env, "config", None), "vehicle_speed_kmh", 0.0) or 0.0)
    phi_ep = EpisodeStats.from_observations(observations, dist=dist, speed_kmh=speed)
    metrics = EpisodeMetrics()
    income: Dict[int, float] = {did: 0.0 for did in observations}

    done = False
    steps = 0
    while not done:
        if max_steps is not None and steps >= max_steps:
            break
        phi_step = GlobalStats.from_observations(observations, dist=phi_ep.dist)
        # ONE LLM query for the whole step (cached inside the controller). The
        # online controller keeps its historical (obs, phi, pref, dist) surface.
        controller.begin_step(observations, phi_step, pref, dist)

        budget = FairnessBudget(strength=pref["fairness"])
        betas = budget.budgets(income)
        bids, _classes = compute_bids(
            observations=observations,
            skills=skills,
            combiner=controller,
            phi_ep=phi_ep,
            phi_step=phi_step,
            budgets=betas,
            w=None,
            temperature=temperature,
            top_k=top_k,
        )
        actions = {did: {"orders": oids} for did, oids in bids.items()}
        observations, rewards, dones, info = env.step(actions)
        for did, r in rewards.items():
            income[did] = income.get(did, 0.0) + float(r)
        metrics.update(rewards, info)
        done = dones["__all__"]
        steps += 1

    return metrics.finalize(total_orders=len(env._all_orders)), controller.meter
