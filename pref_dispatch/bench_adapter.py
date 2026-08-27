"""Adapt the preference-dispatch stack to the benchmark dispatcher contract.

The benchmark harness (``benchmark/evaluate.py`` -> ``run_episode``) drives every
method through a uniform interface: ``act(observations) -> {driver_id:
{"orders": [...]}}``, built by a *factory* ``cfg -> dispatcher`` so each test set
constructs the method against its own :class:`BenchmarkConfig`. Our paradigm-B
method -- :class:`pref_dispatch.evaluate.DispatchController` -- instead takes
``act(observations, pref, income, dist)``: a fixed platform preference ``w``, the
running per-driver income the fairness budget consumes, and a travel-time closure
over the env's road network.

:class:`PrefDispatchAdapter` closes that gap without touching either stack:

* ``pref`` and ``dist`` are captured at construction (the preference is fixed for
  a run; ``dist`` is a stateless query over the env network, identical across env
  instances built from the same config);
* ``income`` is accumulated across steps from the reward stream via the optional
  ``observe_rewards`` hook that ``run_episode`` probes after each ``env.step``.

Because ``make_nyc_env`` and the benchmark share ``make_benchmark_env``, a given
``(regime, split, seed)`` yields byte-identical orders for every method, so
wrapping the controller here makes pref_dispatch directly comparable to the MARL
baselines on one KPI table -- the only obstacle was this interface mismatch, not
the environment.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import os

from benchmark.config import BenchmarkConfig, _make_network
from pref_dispatch.combiner import SingleSkillCombiner
from pref_dispatch.evaluate import DispatchController
from pref_dispatch.global_stats import EpisodeStats
from pref_dispatch.llm.basis import load_basis, load_frozen_combiner
from pref_dispatch.matching import DEFAULT_BLEND_K, DEFAULT_TOP_K
from pref_dispatch.nyc_env import SPLITS_DIR, prev_window_orders
from pref_dispatch.preference import Preference
from pref_dispatch.skills import Skill
from pref_dispatch.wage import driver_wage_from_event


RewardFn = Callable[[Dict], float]


class PrefDispatchAdapter:
    """Wrap a :class:`DispatchController` as a benchmark ``act(observations)``.

    Parameters
    ----------
    controller:
        The frozen paradigm-B controller (frozen skills + frozen combiner).
    pref:
        The fixed platform preference ``w`` for this run.
    dist:
        Travel-time closure over the env road network (from :func:`_make_dist`).
        The road network is deterministic given the benchmark config, so a
        closure built from any env instance of that config answers identically.
    speed_kmh:
        Driver speed carried on the episode-static ``phi_ep`` (final-version
        two-layer stats); read from the benchmark config.
    reward_fn:
        The episode objective ``w`` (LLM-authored callable reward fn) carried on
        ``phi_ep`` and read by the combiner + repositioner. ``None`` = objective-
        blind (legacy behaviour), matching the pre-redesign adapter.
    objective_label:
        Human-readable objective brief for logging / NL explanations.
    """

    def __init__(
        self,
        controller: DispatchController,
        pref: Preference,
        dist: Callable,
        *,
        speed_kmh: float = 0.0,
        reward_fn: Optional[RewardFn] = None,
        objective_label: str = "",
        prev_orders: tuple = (),
    ):
        self.controller = controller
        self.pref = pref
        self.dist = dist
        self.speed_kmh = float(speed_kmh)
        self.reward_fn = reward_fn
        self.objective_label = objective_label
        # PREVIOUS-window orders, turned into the OD matrix on phi_ep at the first
        # act(). Resolved from the config by ``_prev_orders_for_cfg``; empty tuple
        # when this config has no earlier window (then the OD matrix is empty).
        self.prev_orders = tuple(prev_orders)
        # Episode-static phi_ep, built lazily from the FIRST observation the
        # benchmark hands us (the harness drives act(obs) without exposing reset,
        # so the first act() call is the episode start). Computed once, never again.
        self.phi_ep: Optional[EpisodeStats] = None
        # Running per-driver cumulative income (shaping reward), kept for any
        # reward-based diagnostics. Lazily seeded from the first observation.
        self.income: Dict[int, float] = {}
        # Running per-driver cumulative WAGE (take-home fare) -- what the
        # fairness budget equalizes. Accumulated via ``observe_events``.
        self.wage: Dict[int, float] = {}

    def act(self, observations: Dict[int, Dict]) -> Dict[int, Dict]:
        if self.phi_ep is None:
            # First step of the episode: freeze the static scenario descriptor
            # (fleet/layout/scale/dist + objective w). dist lives on phi_ep now.
            self.phi_ep = EpisodeStats.from_observations(
                observations,
                dist=self.dist,
                speed_kmh=self.speed_kmh,
                reward_fn=self.reward_fn,
                objective_label=self.objective_label,
                prev_orders=self.prev_orders,
            )
        if not self.income:
            self.income = {did: 0.0 for did in observations}
        if not self.wage:
            self.wage = {did: 0.0 for did in observations}
        # DispatchController.act already returns {did: {"orders": [...]}}, the
        # exact benchmark action shape -- no translation needed. The budget
        # consumes cumulative WAGE (money), not the shaping reward. phi_step +
        # kappa + w are derived inside act() from phi_ep each step.
        return self.controller.act(
            observations, self.pref, self.income, self.phi_ep,
            fairness_income=self.wage,
        )

    def observe_rewards(self, rewards: Dict[int, float]) -> None:
        """Accumulate per-step shaping reward into cumulative income.

        ``run_episode`` calls this (via getattr probe) after each ``env.step``.
        Retained for reward-based diagnostics; the fairness budget itself now
        consumes WAGE (see :meth:`observe_events`), not this reward income.
        """
        for did, r in rewards.items():
            self.income[did] = self.income.get(did, 0.0) + float(r)

    def observe_events(self, events: Dict[int, Dict]) -> None:
        """Accumulate per-driver WAGE (fare) from the step's event log.

        ``run_episode`` calls this (via getattr probe) after each ``env.step``
        with ``info["events"]`` -- the per-order fields the reward alone does
        not expose. With ``pref["fairness"] == 0`` the budget ignores wage
        entirely, so the efficiency-only main table is unaffected by wage
        timing; feeding it keeps any fairness>0 run rigorous.
        """
        for did, ev in events.items():
            self.wage[did] = self.wage.get(did, 0.0) + driver_wage_from_event(ev)


def _make_dist_closure(cfg: BenchmarkConfig) -> Callable:
    """Build a travel-time closure over THIS config's road network.

    Builds only the road network (not a full env with 1000 drivers + the order
    stream): the ``dist`` closure needs just ``network.shortest_path(a,b).
    travel_time``. The OSMnx backend is process-cached, so this does not re-pay
    graph loading.
    """
    network = _make_network(cfg)

    def dist(a, b) -> float:
        return network.shortest_path(a, b).travel_time

    return dist


def _prev_orders_for_cfg(cfg: BenchmarkConfig) -> tuple:
    """Previous-window orders for THIS benchmark config (``()`` when unavailable).

    The benchmark harness hands the factory a config, not an env, so the OD prior
    is resolved from ``nyc_order_path`` + ``nyc_split`` here instead of from the
    ``prev_window_orders`` stamp :func:`pref_dispatch.evaluate.rollout` reads. Same
    window either way -- always an hour EARLIER than the one being replayed.
    """
    path = getattr(cfg, "nyc_order_path", None)
    if not path:
        return ()
    try:
        return prev_window_orders(
            os.path.relpath(path, SPLITS_DIR), getattr(cfg, "nyc_split", "test")
        )
    except Exception:               # abstract network / hand-made config: no prior
        return ()


def make_pref_factory(
    pref: Preference,
    combiner_name: Optional[str] = None,
    include_evolved: bool = True,
    top_k: int = DEFAULT_TOP_K,
    skill_dir_override: Optional[Sequence[str]] = None,
    combiner_dir_override: Optional[str] = None,
    repositioner=None,
    reward_fn: Optional[RewardFn] = None,
    objective_label: str = "",
    blend_k: int = DEFAULT_BLEND_K,
) -> Callable[[BenchmarkConfig], PrefDispatchAdapter]:
    """Return a benchmark method factory for pref_dispatch under ``pref``.

    The factory loads the frozen basis + frozen Phase-2 combiner, builds a
    :class:`DispatchController`, and wraps it with a ``dist`` closure over an env
    built from THIS test set's config (so the road network matches the scenario
    every other method sees). ``combiner_name=None`` selects the single frozen
    combiner on disk.

    §Phase-3 fine-tune overrides: ``skill_dir_override`` layers fine-tuned skill
    directories on top of the generalist basis (same-name overrides win, via
    :func:`load_basis`'s ``extra_skill_dirs``); ``combiner_dir_override`` points
    :func:`load_frozen_combiner` at a fine-tuned combiners directory instead of the
    default. Both default to ``None`` (unchanged generalist behaviour).

    §Feature-3 reposition: ``repositioner`` is the single self-contained handle
    for idle-driver repositioning -- a :class:`~pref_dispatch.reposition.Repositioner`
    (``scores_fn`` = an evolved Feature-3 scorer, else the demand-gravity
    heuristic). ``None`` (default) means repositioning is OFF: no relocate action
    is ever emitted and the controller is byte-identical to legacy.

    §Final-version objective conditioning: ``reward_fn`` is the episode objective
    ``w`` (a callable reward the combiner + repositioner READ off ``phi_ep`` to
    self-derive their strategy) and ``objective_label`` is its human brief. Default
    ``None`` = objective-blind (legacy balanced behaviour). For the MARL-anchor
    head-to-head, pass the anchor's own default reward so the objective-reading
    stack is graded on -- and reads -- the exact objective the env scores by.

    ``blend_k`` is how many skills may share one driver's decision (v6 item 8);
    ``1`` is the pre-v6 one-hot select and reproduces archived results exactly.
    """

    def factory(cfg: BenchmarkConfig) -> PrefDispatchAdapter:
        skills, _cards = load_basis(
            include_evolved=include_evolved, extra_skill_dirs=skill_dir_override
        )
        combiner_kw = {"skill_names": tuple(skills)}
        if combiner_dir_override is not None:
            combiner_kw["combiners_dir"] = combiner_dir_override
        combiner, _meta = load_frozen_combiner(combiner_name, **combiner_kw)
        # TRAIN/TEST OPERATING-POINT CHECK (2026-08-10). A combiner frozen since
        # this date records the ``top_k`` / ``blend_k`` it was evolved under. If
        # evaluation runs at different values the policy is being graded off its
        # training distribution -- that is how the gate spent months at top_k=20
        # against a stack trained at 60 -- so say so loudly rather than silently
        # producing numbers. Older artifacts record nothing and are skipped.
        for key, used in (("train_top_k", top_k), ("train_blend_k", blend_k)):
            trained = (_meta or {}).get(key)
            if trained is not None and int(trained) != int(used):
                print(f"[bench_adapter] WARNING: {combiner_name or 'combiner'} was "
                      f"TRAINED at {key.replace('train_', '')}={int(trained)} but is "
                      f"being EVALUATED at {int(used)}. Train/test mismatch -- the "
                      f"result is off-distribution.")
        if hasattr(combiner, "blend_k"):
            combiner.blend_k = int(blend_k)
        controller = DispatchController(
            combiner,
            skills=skills,
            top_k=top_k,
            repositioner=repositioner,
            blend_k=blend_k,
        )
        return PrefDispatchAdapter(
            controller, pref, _make_dist_closure(cfg), speed_kmh=cfg.speed_kmh,
            reward_fn=reward_fn, objective_label=objective_label,
            prev_orders=_prev_orders_for_cfg(cfg),
        )

    return factory


def make_single_skill_factory(
    skill_name: str,
    *,
    skill: Optional[Skill] = None,
    top_k: int = DEFAULT_TOP_K,
) -> Callable[[BenchmarkConfig], PrefDispatchAdapter]:
    """Return a benchmark factory that dispatches the WHOLE fleet with one skill.

    This is the weakest rung of the effectiveness ladder (proposal step 4): every
    driver always uses ``skill_name`` via :class:`SingleSkillCombiner`, with no
    upper combiner and no preference conditioning. It is the single-skill
    counterpart to :func:`make_pref_factory` and shares the exact same benchmark
    contract, so a frozen basis skill or a pure-Phase-1 evolved scorer can be
    dropped straight into the head-to-head table beside the two-layer method,
    the MARL checkpoints and the heuristics.

    ``skill_name`` must resolve inside :func:`load_basis` (a handwritten seed or
    an evolved skill under ``pref_dispatch/evolved/skills/``). To score a scorer
    that is NOT in the frozen basis (e.g. a freshly evolved pure-Phase-1 skill),
    pass the compiled :class:`Skill` object directly via ``skill``.

    The preference is irrelevant to a single skill (``SingleSkillCombiner``
    ignores it), so a neutral efficiency preference is used; ``income`` timing is
    likewise moot (fairness is off).
    """
    neutral = Preference(
        weights={"revenue": 0.5, "service": 0.5, "fairness": 0.0}
    )

    def factory(cfg: BenchmarkConfig) -> PrefDispatchAdapter:
        if skill is not None:
            skills = {skill_name: skill}
        else:
            basis, _cards = load_basis(include_evolved=True)
            if skill_name not in basis:
                raise KeyError(
                    f"skill {skill_name!r} not in frozen basis "
                    f"{sorted(basis)}; pass skill= to score an out-of-basis one."
                )
            skills = {skill_name: basis[skill_name]}
        controller = DispatchController(
            SingleSkillCombiner(skill_name), skills=skills, top_k=top_k
        )
        return PrefDispatchAdapter(
            controller, neutral, _make_dist_closure(cfg), speed_kmh=cfg.speed_kmh,
            prev_orders=_prev_orders_for_cfg(cfg),
        )

    return factory
