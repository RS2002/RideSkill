"""Rollout loop: given preference + objective ``w`` -> episode metrics.

:class:`DispatchController` wires the full closed loop for one episode:

    obs_0 -> EpisodeStats(phi_ep)          # ONCE, episode-static (fleet, layout,
                                           #   map scale, dist, objective w)
    each step:
      obs -> GlobalStats(phi_step)         # live: pending/idle, demand, kappa
          -> FairnessBudget(income) -> beta_d
          -> compute_bids(combiner picks/blends skills using phi_ep/phi_step/w;
                          softmax; x beta)
          -> Repositioner overlay (idle drivers, phi_ep/phi_step/kappa/w)
          -> env.step(bids) -> rewards, info
          -> metrics.update(...)

Two-layer stats (final-version redesign): ``phi_ep`` is computed once at reset and
never again (episode-static, no future-order leakage); ``phi_step`` is recomputed
each step (live state + per-region kappa). The travel-time closure ``dist`` wraps
the *env's own* road network and lives on ``phi_ep`` (the network is fixed for the
episode), so skills/combiner/repositioner take it via ``phi_ep.dist`` rather than
as a separate argument.

``w`` (:attr:`EpisodeStats.reward_fn`) is the episode objective as an LLM-authored
callable reward function (``None`` = objective-blind). Only the combiner and the
repositioner read it; skills stay objective-specialist. Translating an NL objective
into ``w`` is an offline, once-per-episode LLM step done by the caller (see
:mod:`pref_dispatch.llm.objective`); the core loop just carries the resulting
callable on ``phi_ep``.

The controller is deterministic given the env seed, the preference, and ``w``, so
the same inputs always yield the same metrics -- the property the M2/M3 evolution
loop relies on when it uses episode metrics as fitness.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from ride_gym import RidePoolEnv

from pref_dispatch.budget import FairnessBudget
from pref_dispatch.combiner import Combiner
from pref_dispatch.global_stats import EpisodeStats, GlobalStats
from pref_dispatch.matching import DEFAULT_BLEND_K, DEFAULT_TOP_K, compute_bids
from pref_dispatch.metrics import EpisodeMetrics
from pref_dispatch.preference import Preference
from pref_dispatch.reposition import Repositioner
from pref_dispatch.skills import Skill, default_skill_basis
from pref_dispatch.wage import driver_wage_from_event

RewardFn = Callable[[Dict], float]


class DispatchController:
    """Runs the per-step dispatch decision for a fixed episode."""

    def __init__(
        self,
        combiner: Combiner,
        skills: Optional[Dict[str, Skill]] = None,
        temperature: float = 1.0,
        top_k: int = DEFAULT_TOP_K,
        repositioner: Optional[Repositioner] = None,
        blend_k: Optional[int] = None,
    ):
        if skills is None:
            skills = {s.name: s for s in default_skill_basis()}
        self.skills = skills
        self.combiner = combiner
        self.temperature = temperature
        self.top_k = top_k
        # How many skills may share one driver's decision (v6 item 8). ``None``
        # (default) takes the combiner's own setting so the knob lives in ONE
        # place -- an LLMCombiner built with blend_k=3 is blended with 3 here
        # without every call site having to repeat it. Combiners that do not
        # expose the attribute (single-skill, heuristic) fall back to 1, the
        # pre-v6 one-hot behaviour.
        self.blend_k = (
            int(getattr(combiner, "blend_k", DEFAULT_BLEND_K))
            if blend_k is None else int(blend_k)
        )
        # Idle-driver repositioning. ``None`` (default) = off: no relocate action
        # is ever emitted. A :class:`~pref_dispatch.reposition.Repositioner` is the
        # single, self-contained handle for the feature -- independent of the
        # order-scoring skill stack.
        self.repositioner = repositioner

    def act(
        self,
        observations: Dict[int, Dict],
        pref: Preference,
        income: Dict[int, float],
        phi_ep: EpisodeStats,
        fairness_income: Optional[Dict[int, float]] = None,
    ) -> Dict[int, Dict]:
        # phi_step: live per-step state + kappa (recomputed every step). The
        # travel-time closure comes from the episode-static phi_ep.
        phi_step = GlobalStats.from_observations(observations, dist=phi_ep.dist)
        w = phi_ep.reward_fn
        budget = FairnessBudget(strength=pref["fairness"])
        # The budget equalizes driver WAGE (take-home fare), not the shaping
        # reward: pass ``fairness_income`` (cumulative wage) when available and
        # fall back to ``income`` (cumulative reward) only for legacy callers
        # that do not track wage -- keeping old behaviour byte-for-byte.
        betas = budget.budgets(fairness_income if fairness_income is not None else income)
        bids, _classes = compute_bids(
            observations=observations,
            skills=self.skills,
            combiner=self.combiner,
            phi_ep=phi_ep,
            phi_step=phi_step,
            budgets=betas,
            w=w,
            temperature=self.temperature,
            top_k=self.top_k,
            blend_k=self.blend_k,
        )
        # Translate bid sets into env actions. Empty bid == keep current state.
        actions = {did: {"orders": oids} for did, oids in bids.items()}
        if self.repositioner is None:
            return actions  # repositioning off: only order bids are emitted.

        # Repositioning on: idle drivers with an empty bid may cruise toward a
        # high-demand neighbour region instead of sitting still. The Repositioner
        # only returns eligible (idle + empty-bid) drivers, so overlaying its
        # relocate actions can never violate the env's mutual-exclusion rule.
        # ``betas`` go along: the scorer is trained across fairness strengths and
        # needs to know which cars the matcher is currently favouring, not just how
        # hard it is favouring them (phi_ep.fairness_strength).
        for did, region_idx in self.repositioner.targets(
            observations, bids, phi_ep, phi_step, w=w, budgets=betas
        ).items():
            actions[did] = {"relocate": int(region_idx)}
        return actions


def _make_dist(env: RidePoolEnv) -> Callable:
    """Travel-time estimator backed by the env's road network."""

    def dist(a, b) -> float:
        return env.network.shortest_path(a, b).travel_time

    return dist


def rollout(
    env: RidePoolEnv,
    controller: DispatchController,
    pref: Preference,
    seed: Optional[int] = None,
    *,
    reward_fn: Optional[RewardFn] = None,
    objective_label: str = "",
) -> Dict[str, float]:
    """Run one full episode under ``pref`` and return finalized metrics.

    ``reward_fn`` is the episode objective ``w`` (LLM-authored callable, translated
    once offline by the caller); it is carried on ``phi_ep`` and read only by the
    combiner + repositioner. ``None`` runs objective-blind (legacy behaviour).
    """
    observations, _info = env.reset(seed=seed)
    dist = _make_dist(env)

    # phi_ep: episode-static scenario descriptor. Computed ONCE, never recomputed.
    speed = float(getattr(getattr(env, "config", None), "vehicle_speed_kmh", 0.0) or 0.0)
    phi_ep = EpisodeStats.from_observations(
        observations,
        dist=dist,
        speed_kmh=speed,
        reward_fn=reward_fn,
        objective_label=objective_label,
        # The repositioner is trained ACROSS fairness strengths and has to know
        # which one this episode runs at (the matcher multiplies the combiner's
        # scores by the wage-equalising budget at exactly this strength). The
        # combiner ignores the field.
        fairness_strength=float(pref.get("fairness", 0.0)),
        # OD matrix over the PREVIOUS window. The env is stamped with those orders
        # at construction (``nyc_env.stamp_prev_window``, applied by both
        # ``make_nyc_env`` and ``scenario.build_env``), so no caller has to thread
        # the window through. Envs built by other paths simply carry nothing and
        # get an empty OD matrix. Always an EARLIER hour than the one replayed.
        prev_orders=getattr(env, "prev_window_orders", ()),
    )

    metrics = EpisodeMetrics()
    income: Dict[int, float] = {did: 0.0 for did in observations}
    # Cumulative per-driver WAGE (take-home fare) -- what the fairness budget
    # equalizes. Kept separate from ``income`` (cumulative shaping reward, which
    # metrics still consumes), so fairness acts on money, not penalties.
    wage: Dict[int, float] = {did: 0.0 for did in observations}

    done = False
    while not done:
        actions = controller.act(
            observations, pref, income, phi_ep, fairness_income=wage
        )
        observations, rewards, dones, info = env.step(actions)
        # Keep the online income (metrics) and wage (fairness budget) in sync.
        for did, r in rewards.items():
            income[did] = income.get(did, 0.0) + float(r)
        for did, ev in info.get("events", {}).items():
            wage[did] = wage.get(did, 0.0) + driver_wage_from_event(ev)
        # Post-step driver status (idle/to_pickup/relocating/...) lets the metrics
        # attribute empty moves made while RELOCATING to repositioning. With
        # reposition off no driver is ever "relocating", so the KPI stays 0.
        driver_status = {
            did: o["self"]["status"] for did, o in observations.items()
        }
        metrics.update(rewards, info, driver_status=driver_status)
        done = dones["__all__"]

    return metrics.finalize(total_orders=len(env._all_orders))
