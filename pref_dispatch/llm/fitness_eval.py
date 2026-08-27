"""Phase-1 skill fitness: roll out ONE candidate skill and score it.

Phase 1 discovers lower-layer skills by quality-diversity search (§4). To rank
candidates under a skill's *own* LLM-authored fitness we need a deterministic
map ``skill -> scalar``. This module provides it:

    skill --(single-skill dispatch on the real Manhattan env)--> episode metrics
          --(the skill's self-authored fitness(metrics))--------> scalar

"Single-skill dispatch" means every driver uses this one skill (via
:class:`~pref_dispatch.combiner.SingleSkillCombiner`), so the metrics reflect
that skill's behaviour in isolation -- the right signal for evolving a
single-objective specialist. We reuse the existing, tested
:func:`pref_dispatch.evaluate.rollout` rather than re-implementing the loop.

Multi-regime (§4.8): a skill is evaluated on several time-of-day regimes on the
TRAIN split and the per-regime fitnesses are averaged, so an evolved skill must
generalise across demand levels rather than overfit one hour. Metrics scales
differ a lot across regimes (peak revenue >> off-peak), so raw fitness values are
not comparable across regimes; we therefore also report each regime's fitness
*relative to a reference skill* (the handwritten seed) -- a scale-free Δ that the
QD loop can average honestly. The absolute mean is kept too for logging.

Feature 3 adds :func:`evaluate_repositioner` -- a SEPARATE fitness path for the
idle-driver reposition scorer (LLM-authored per-region base scores wrapped in a
:class:`~pref_dispatch.reposition.Repositioner`). It is a fixed-fitness arm, so it
does not reuse the skill machinery; it rolls the SAME scenario once with the
scorer's repositioning ON and once OFF, and the fitness is the Δ of a fixed
service-oriented formula -- the exact "improvement over not repositioning" the
Feature-3 prompt promises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from pref_dispatch.combiner import SingleSkillCombiner
from pref_dispatch.evaluate import DispatchController, rollout
from pref_dispatch.nyc_env import make_nyc_env
from pref_dispatch.preference import Preference
from pref_dispatch.reposition import Repositioner
from pref_dispatch.scenario import Scenario, ScenarioSampler, build_env
from pref_dispatch.skills import Skill

# Efficiency-only preference (fairness deferred: FairnessBudget(strength=0) is
# the identity, all beta=1.0). Phase-1 skills are single-objective specialists,
# so the preference passed to the loop is neutral; the skill itself encodes the
# objective. Kept explicit so the contract is visible.
_NEUTRAL_PREF = Preference(weights={"revenue": 0.5, "service": 0.5, "fairness": 0.0})

# Default train regimes to average over (§6 hyperparameters).
DEFAULT_REGIMES: Sequence[str] = ("offpeak", "shoulder", "peak")

# Evaluation operating point (§6). The deployment benchmark uses ~1000 drivers.
# We evaluate the FULL real-hour demand against a slightly reduced fleet so the
# single-objective trade-offs stay exposed (taking order A means declining order
# B) WITHOUT collapsing to an unrealistically empty system. 800 drivers on the
# real peak window gives service_rate ~0.71 -- a realistic, reviewable density
# -- while the 800-car full-demand tension calibration still shows the optimal
# skill flipping with the preference (pick_diversity=2, fixed_gap 0.13/0.20/0.24
# across offpeak/shoulder/peak; see cache/logs/calib_800_full.log), so fitness
# still separates candidates. An earlier 150-car scarcity point discriminated
# too but at an unrealistically low service_rate (~0.13); 800 keeps the tension
# and adds realism. ~4.9s/rollout at 800 (vs ~2s at 150). The frozen skills /
# Phase-2 combiner and the head-to-head tables all share THIS operating point so
# nothing is evaluated out-of-distribution from where it evolved.
EVAL_NUM_DRIVERS: int = 800
EVAL_ORDER_LIMIT: Optional[int] = None  # full real hour

SkillLike = Skill  # CompiledSkill duck-types as Skill (.score / .noop_score).
FitnessFn = Callable[[Dict[str, float]], float]


@dataclass
class SkillEval:
    """Result of evaluating one skill under one fitness across regimes."""

    fitness_mean: float  # mean absolute fitness over regimes
    per_regime_fitness: Dict[str, float] = field(default_factory=dict)
    per_regime_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Present only when a reference skill was supplied: mean of per-regime
    # (candidate_fitness - reference_fitness). Scale-free; the QD selection key.
    delta_vs_reference: Optional[float] = None
    per_regime_delta: Dict[str, float] = field(default_factory=dict)


def rollout_skill_metrics(
    skill: SkillLike,
    regime: str,
    split: str = "train",
    *,
    seed: int = 0,
    num_drivers: int = EVAL_NUM_DRIVERS,
    driver_capacity: int = 4,
    order_limit: Optional[int] = EVAL_ORDER_LIMIT,
) -> Dict[str, float]:
    """Single-skill dispatch on one real-Manhattan regime -> episode metrics.

    Defaults evaluate the FULL real-hour demand against a reduced fleet
    (:data:`EVAL_NUM_DRIVERS`) so supply scarcity exposes the skill's objective
    trade-off (see the module note). Pass ``num_drivers=1000`` /
    ``order_limit=...`` to score at the deployment operating point instead.
    """
    env = make_nyc_env(
        seed=seed,
        regime=regime,
        split=split,
        num_drivers=num_drivers,
        driver_capacity=driver_capacity,
        order_limit=order_limit,
    )
    name = getattr(skill, "name", "candidate")
    controller = DispatchController(
        combiner=SingleSkillCombiner(name),
        skills={name: skill},
    )
    return rollout(env, controller, _NEUTRAL_PREF, seed=seed)


def evaluate_skill(
    skill: SkillLike,
    fitness_fn: FitnessFn,
    *,
    regimes: Sequence[str] = DEFAULT_REGIMES,
    split: str = "train",
    seed: int = 0,
    num_drivers: int = EVAL_NUM_DRIVERS,
    order_limit: Optional[int] = EVAL_ORDER_LIMIT,
    reference: Optional[SkillLike] = None,
    reference_metrics: Optional[Dict[str, Dict[str, float]]] = None,
) -> SkillEval:
    """Evaluate ``skill`` under its ``fitness_fn`` across ``regimes``.

    When ``reference`` is given, also reports the scale-free
    ``delta_vs_reference`` (mean over regimes of candidate - reference fitness),
    which is what the QD loop should compare across regimes/candidates.

    ``reference_metrics`` (regime -> metrics dict) lets the caller pass in the
    reference seed's already-computed metrics so it is not re-rolled for every
    candidate (the reference is deterministic for a fixed seed/fleet/regime).
    Use :func:`reference_metrics_for` to build it once per run.
    """
    per_fit: Dict[str, float] = {}
    per_met: Dict[str, Dict[str, float]] = {}
    per_delta: Dict[str, float] = {}

    for regime in regimes:
        metrics = rollout_skill_metrics(
            skill, regime, split, seed=seed,
            num_drivers=num_drivers, order_limit=order_limit,
        )
        per_met[regime] = metrics
        per_fit[regime] = float(fitness_fn(metrics))

        ref_met = None
        if reference_metrics is not None:
            ref_met = reference_metrics.get(regime)
        elif reference is not None:
            ref_met = rollout_skill_metrics(
                reference, regime, split, seed=seed,
                num_drivers=num_drivers, order_limit=order_limit,
            )
        if ref_met is not None:
            per_delta[regime] = per_fit[regime] - float(fitness_fn(ref_met))

    fitness_mean = sum(per_fit.values()) / len(per_fit) if per_fit else 0.0
    delta_mean = (
        sum(per_delta.values()) / len(per_delta) if per_delta else None
    )
    return SkillEval(
        fitness_mean=fitness_mean,
        per_regime_fitness=per_fit,
        per_regime_metrics=per_met,
        delta_vs_reference=delta_mean,
        per_regime_delta=per_delta,
    )


def reference_metrics_for(
    reference: SkillLike,
    *,
    regimes: Sequence[str] = DEFAULT_REGIMES,
    split: str = "train",
    seed: int = 0,
    num_drivers: int = EVAL_NUM_DRIVERS,
    order_limit: Optional[int] = EVAL_ORDER_LIMIT,
) -> Dict[str, Dict[str, float]]:
    """Roll out the reference seed once per regime -> ``{regime: metrics}``.

    The reference is deterministic for a fixed seed/fleet/regime, so computing it
    once and passing it to every :func:`evaluate_skill` call halves the rollouts
    in an evolution run.
    """
    return {
        regime: rollout_skill_metrics(
            reference, regime, split, seed=seed,
            num_drivers=num_drivers, order_limit=order_limit,
        )
        for regime in regimes
    }


# =========================================================================== #
# v2: random-scenario fitness with scale rescaling.                           #
# =========================================================================== #
# v1 evaluated at ONE fixed operating point (800 drivers, cap 4, three fixed
# regimes). v2 instead samples a fresh batch of scenarios per evaluation from a
# domain-randomization envelope (fleet/capacity/speed/time-of-day/preference), so
# an evolved skill is selected for how well it GENERALIZES rather than how well it
# overfits one point. Because reward magnitudes differ by orders of magnitude
# across scenarios (a 2000-driver peak hour earns far more than a 100-driver
# off-peak one), raw fitness is not comparable across scenarios; we rescale each
# scenario's fitness to a scale-free number before averaging (see RESCALE_MODES).


@dataclass
class ScenarioEval:
    """Result of evaluating one skill over a random batch of scenarios."""

    score: float                          # aggregate (mean of rescaled per-scenario)
    per_scenario_score: List[float] = field(default_factory=list)   # rescaled
    per_scenario_raw: List[float] = field(default_factory=list)     # pre-rescale
    per_scenario_metrics: List[Dict[str, float]] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)


def rollout_skill_on_scenario(
    skill: SkillLike,
    scenario: Scenario,
) -> Dict[str, float]:
    """Single-skill dispatch on one concrete :class:`Scenario` -> episode metrics.

    The env (fleet/capacity/speed/window, party clamped to capacity) is built via
    :func:`pref_dispatch.scenario.build_env`, and the rollout is scored under the
    SCENARIO's own preference so the randomized reward-preference axis is exercised
    too. A neutral preference is used for single-skill specialists (the skill, not
    the preference, encodes the objective), matching :func:`rollout_skill_metrics`.
    """
    env = build_env(scenario)
    name = getattr(skill, "name", "candidate")
    controller = DispatchController(
        combiner=SingleSkillCombiner(name),
        skills={name: skill},
    )
    return rollout(env, controller, _NEUTRAL_PREF, seed=scenario.seed)


# Scale factors used to turn a raw per-scenario fitness into a scale-free number.
def _fleet_orders_scale(scenario: Scenario, metrics: Dict[str, float]) -> float:
    """Rough reward scale of a scenario: fleet size x served orders (+1 guard)."""
    served = float(metrics.get("assigned", 0.0)) or 1.0
    return max(1.0, float(scenario.num_drivers) * served)


def evaluate_skill_random_scenarios(
    skill: SkillLike,
    fitness_fn: FitnessFn,
    scenarios: Sequence[Scenario],
    *,
    rescale: str = "reference",
    reference: Optional[SkillLike] = None,
    reference_metrics: Optional[Sequence[Dict[str, float]]] = None,
) -> ScenarioEval:
    """Evaluate ``skill`` under ``fitness_fn`` over a batch of random scenarios.

    ``scenarios`` is a concrete list (drawn once by the caller via
    :class:`ScenarioSampler`) so the SAME batch can be reused across candidates in
    one comparison round -- fair "who is better", while the batch still changes
    across rounds to keep pushing generalization.

    ``rescale`` selects how each scenario's raw fitness becomes scale-free before
    averaging:

    * ``"reference"`` (default, most honest): score = candidate_fitness -
      reference_fitness on the SAME scenario (a scale-free Δ vs a fixed baseline
      skill). Provide ``reference`` (rolled per scenario) or precomputed
      ``reference_metrics`` (one dict per scenario, same order).
    * ``"fleet_orders"``: score = raw_fitness / (num_drivers x served_orders) --
      no baseline rollouts, cheaper, but a coarser normaliser.
    * ``"none"``: raw fitness (only sensible when the batch is a single scale).
    """
    per_raw: List[float] = []
    per_score: List[float] = []
    per_met: List[Dict[str, float]] = []
    labels: List[str] = []

    for i, sc in enumerate(scenarios):
        metrics = rollout_skill_on_scenario(skill, sc)
        raw = float(fitness_fn(metrics))
        per_raw.append(raw)
        per_met.append(metrics)
        labels.append(sc.label())

        if rescale == "reference":
            ref_met = None
            if reference_metrics is not None and i < len(reference_metrics):
                ref_met = reference_metrics[i]
            elif reference is not None:
                ref_met = rollout_skill_on_scenario(reference, sc)
            ref_fit = float(fitness_fn(ref_met)) if ref_met is not None else 0.0
            per_score.append(raw - ref_fit)
        elif rescale == "fleet_orders":
            per_score.append(raw / _fleet_orders_scale(sc, metrics))
        elif rescale == "none":
            per_score.append(raw)
        else:
            raise ValueError(f"unknown rescale mode {rescale!r}")

    score = sum(per_score) / len(per_score) if per_score else 0.0
    return ScenarioEval(
        score=score,
        per_scenario_score=per_score,
        per_scenario_raw=per_raw,
        per_scenario_metrics=per_met,
        labels=labels,
    )


def reference_metrics_for_scenarios(
    reference: SkillLike,
    scenarios: Sequence[Scenario],
) -> List[Dict[str, float]]:
    """Roll the reference skill once per scenario -> per-scenario metrics list.

    Deterministic for a fixed scenario (its env seed is pinned), so computing it
    once and passing it to every :func:`evaluate_skill_random_scenarios` call in a
    round halves the rollouts under ``rescale="reference"``.
    """
    return [rollout_skill_on_scenario(reference, sc) for sc in scenarios]


# =========================================================================== #
# Feature 3: fixed-fitness repositioner evaluation (Repositioner ON vs OFF).   #
# =========================================================================== #
# Repositioning is REWARD-FREE in this environment (the benchmark sets
# ``empty_move_penalty == idle_penalty == 0``), so the platform reward cannot
# grade an idle-car policy -- its payoff is in service coverage and waiting time.
# The Feature-3 fitness is therefore FIXED by the researcher as the improvement
# the repositioner brings over repositioning being OFF on the SAME scenario,
# which is exactly what the prompt promises the model.

# Fixed fitness coefficients (see prompts/reposition_evolve.py FIXED_REPOSITION_OBJECTIVE).
_REPOSITION_SERVICE_COEF = 100.0   # lift in service_rate (%), primary reward
_REPOSITION_TIME_COEF = -1.0       # cut in mean_service_time (minutes)
_REPOSITION_WASTE_COEF = -5.0      # share of empty distance spent cruising


def reposition_fitness(metrics: Dict[str, float]) -> float:
    """The FIXED Feature-3 fitness formula over one episode's metrics.

    ``100 * service_rate - mean_service_time - 5 * reposition_distance_ratio``.
    Higher is better; a scorer that cruises cars pointlessly scores BELOW zero
    relative to staying put (it moves empty cars without lifting service).
    """
    return (
        _REPOSITION_SERVICE_COEF * float(metrics.get("service_rate", 0.0))
        - _REPOSITION_TIME_COEF * -1.0 * float(metrics.get("mean_service_time", 0.0))
        - _REPOSITION_WASTE_COEF * -1.0 * float(metrics.get("reposition_distance_ratio", 0.0))
    )


@dataclass
class RepositionerEval:
    """Result of evaluating one reposition scorer on one scenario (ON vs OFF)."""

    fitness: float                 # fixed-fitness Δ: reposition ON - OFF
    metrics_on: Dict[str, float]   # episode metrics with the scorer's repositioning
    metrics_off: Dict[str, float]  # same scenario, repositioning OFF (baseline)
    delta_service_rate: float
    delta_mean_service_time: float
    delta_reposition_ratio: float


def evaluate_repositioner(
    scores_fn: Optional[Callable[[Dict, object, Callable], Dict[int, float]]],
    scenario: Scenario,
    *,
    combiner: Optional[object] = None,
    skills: Optional[Dict[str, Skill]] = None,
    strength: float = 1.0,
    reward_function=None,
    seed: Optional[int] = None,
) -> RepositionerEval:
    """Measure a reposition scorer's fixed fitness on ``scenario`` (ON vs OFF).

    ``scores_fn`` is the LLM-authored per-region scorer to drop into
    ``Repositioner(scores_fn=...)``; ``None`` means the demand-gravity heuristic
    is used (that IS a valid candidate to measure -- it is the frozen baseline
    the LLM arm competes against). Reposition OFF is ``repositioner=None``.

    One episode is rolled with the scorer's repositioning ON and one with it OFF
    on the SAME scenario (same env seed), so the Δ fitness isolates the
    repositioner's effect. The fixed formula
    ``100*service_rate - mean_service_time - 5*reposition_distance_ratio`` is
    applied to each, and the reported ``fitness`` is ON - OFF (positive = the
    repositioner helps; negative = it hurts -- wasted cruising).

    ``reward_function`` (final-version B3): the episode objective. When given it is
    injected into the env AND handed to the repositioner as ``w`` on ``phi_ep``
    (via ``rollout(reward_fn=...)``), so an objective-reading scorer can self-derive
    what a relocation target is worth for THIS objective -- the same reads-w loop
    the combiner uses. ``None`` rolls objective-blind (the reward-free Feature-3
    default, byte-compatible with the legacy path). The fixed service-oriented
    fitness is unchanged regardless -- ``w`` only steers the scorer, it does not
    grade it.

    ``combiner``/``skills`` default to the frozen basis + its single frozen
    combiner when ``None``; pass explicit values to evaluate against a different
    dispatch stack (e.g. the MARL-anchor combiner).
    """
    if combiner is None or skills is None:
        from pref_dispatch.llm.basis import load_basis, load_frozen_combiner

        _skills, _cards = load_basis(include_evolved=True)
        _combiner, _ = load_frozen_combiner(skill_names=tuple(_skills))
        if combiner is None:
            combiner = _combiner
        if skills is None:
            skills = _skills

    env = build_env(scenario, reward_function=reward_function)
    if seed is None:
        seed = scenario.seed
    # The scorer reads the SAME objective the episode carries: bind the env-shape
    # reward into the single-arg ``w(event)`` the repositioner consumes on phi_ep.
    w = None
    if reward_function is not None:
        def w(event: Dict) -> float:  # noqa: E306 -- local objective adapter
            return float(reward_function(0, event))

    # Baseline: repositioning OFF (legacy byte-identical dispatch).
    off_controller = DispatchController(combiner, skills=skills, repositioner=None)
    metrics_off = rollout(env, off_controller, scenario.preference, seed=seed, reward_fn=w)

    # Repositioning ON with this scorer (default strength 1.0 = full aggressiveness).
    on_controller = DispatchController(
        combiner, skills=skills,
        repositioner=Repositioner(strength=strength, scores_fn=scores_fn),
    )
    metrics_on = rollout(env, on_controller, scenario.preference, seed=seed, reward_fn=w)

    fitness = reposition_fitness(metrics_on) - reposition_fitness(metrics_off)
    return RepositionerEval(
        fitness=fitness,
        metrics_on=metrics_on,
        metrics_off=metrics_off,
        delta_service_rate=float(metrics_on.get("service_rate", 0.0))
        - float(metrics_off.get("service_rate", 0.0)),
        delta_mean_service_time=float(metrics_on.get("mean_service_time", 0.0))
        - float(metrics_off.get("mean_service_time", 0.0)),
        delta_reposition_ratio=float(metrics_on.get("reposition_distance_ratio", 0.0))
        - float(metrics_off.get("reposition_distance_ratio", 0.0)),
    )


# =========================================================================== #
# Final version (B3): reward-under-w repositioner fitness (matches Phase 1/2). #
# =========================================================================== #
# The fixed ON-vs-OFF service formula above is the ORIGINAL Feature-3 arm. The
# final version grades the repositioner by the SAME yardstick as the combiner
# (Phase 2): roll the frozen dispatch stack WITH the scorer active under the
# episode objective ``w`` and fairness ``strength``, take the fleet-mean cumulative
# authored reward (``income_mean``), and min-max normalise it in that
# (scenario, objective) pair's single-skill reference frame. Averaging the per-pair
# [0,1] scalars over a batch of sampled objectives+strengths selects ONE frozen
# repositioner that lifts the OBJECTIVE (whatever it is), not a fixed service proxy
# -- exactly the ``scalarize_reward``-on-``income_mean`` fitness Phase 2 uses.


@dataclass
class RepositionerRewardEval:
    """Reward-under-``w`` fitness of a reposition scorer over an objective batch.

    ``fitness`` is the mean over ``zip(scenarios, objectives, strengths)`` of the
    scorer's normalised env reward (``income_mean`` scale-free in each pair's own
    frame). ``per_pair`` maps pair index -> that pair's [0,1] scalar. ``sample_metrics``
    keeps one representative episode's raw metrics for freeze-time provenance.
    """

    fitness: float
    per_pair: Dict[int, float] = field(default_factory=dict)
    mean_income_mean: float = 0.0
    sample_metrics: Optional[Dict[str, float]] = None


def evaluate_repositioner_reward(
    scores_fn: Optional[Callable[[Dict, object, object, object, Optional[Callable]], Dict[int, float]]],
    scenario: Scenario,
    frame,
    *,
    combiner: Optional[object] = None,
    skills: Optional[Dict[str, Skill]] = None,
    strength: float = 1.0,
    reward_function=None,
    seed: Optional[int] = None,
):
    """Roll the dispatch stack with the scorer ON under ``w`` + ``strength`` -> (scalar, metrics).

    A SINGLE rollout (no ON/OFF pair): the repositioner is graded on the objective
    it lifts, not on a service Δ. ``frame`` is that (scenario, objective) pair's
    single-skill env-reward frame (from
    :func:`pref_dispatch.llm.combiner_eval.build_scenario_norm_frame`). Returns the
    normalised ``income_mean`` scalar in ``[0, 1]`` and the raw episode metrics.
    """
    # Deferred import: combiner_eval imports EVAL_NUM_DRIVERS from this module, so a
    # module-level import here would be circular. scalarize_reward is a pure helper.
    from pref_dispatch.llm.combiner_eval import scalarize_reward

    if combiner is None or skills is None:
        from pref_dispatch.llm.basis import load_basis, load_frozen_combiner

        _skills, _cards = load_basis(include_evolved=True)
        _combiner, _ = load_frozen_combiner(skill_names=tuple(_skills))
        if combiner is None:
            combiner = _combiner
        if skills is None:
            skills = _skills

    env = build_env(scenario, reward_function=reward_function)
    if seed is None:
        seed = scenario.seed
    # Bind the episode's authored reward into the single-arg w(event) the scorer reads
    # on phi_ep -- the SAME reads-w loop the combiner uses (driver id 0, objective is
    # driver-agnostic). None => objective-blind (scorer must still work).
    w = None
    if reward_function is not None:
        def w(event: Dict) -> float:  # noqa: E306 -- local objective adapter
            return float(reward_function(0, event))

    controller = DispatchController(
        combiner, skills=skills,
        repositioner=Repositioner(strength=strength, scores_fn=scores_fn),
    )
    metrics = rollout(env, controller, scenario.preference, seed=seed, reward_fn=w)
    return scalarize_reward(metrics, frame), metrics


def evaluate_repositioner_reward_batch(
    scores_fn,
    scenarios: Sequence[Scenario],
    objectives: Sequence[object],
    strengths: Sequence[float],
    frames: Sequence[object],
    *,
    combiner: Optional[object] = None,
    skills: Optional[Dict[str, Skill]] = None,
) -> RepositionerRewardEval:
    """Mean reward-under-``w`` fitness over ``zip(scenarios, objectives, strengths)``.

    Each pair injects its objective's reward into the env AND hands it to the scorer
    as ``w`` (via ``phi_ep``), rolls the stack with the scorer active at that pair's
    fairness ``strength``, and normalises ``income_mean`` in that pair's ``frames[i]``.
    The [0,1] per-pair scalars are averaged -- the SAME Phase-2 fitness, so one frozen
    repositioner best-responds ACROSS objectives and strengths. ``frames`` is built
    once by :func:`pref_dispatch.llm.combiner_eval.build_objective_frames` and shared
    across candidates in a round.
    """
    if not (len(scenarios) == len(objectives) == len(strengths) == len(frames)):
        raise ValueError(
            "scenarios, objectives, strengths, frames must be the same length"
        )
    per_pair: Dict[int, float] = {}
    incomes: List[float] = []
    sample_metrics: Optional[Dict[str, float]] = None
    for i, (sc, obj, strength, frame) in enumerate(
        zip(scenarios, objectives, strengths, frames)
    ):
        scalar, metrics = evaluate_repositioner_reward(
            scores_fn, sc, frame,
            combiner=combiner, skills=skills,
            strength=float(strength),
            reward_function=getattr(obj, "reward_function", None),
        )
        per_pair[i] = scalar
        incomes.append(float(metrics.get("income_mean", 0.0)))
        if sample_metrics is None:
            sample_metrics = metrics
    n = len(per_pair) or 1
    return RepositionerRewardEval(
        fitness=sum(per_pair.values()) / n,
        per_pair=per_pair,
        mean_income_mean=sum(incomes) / n,
        sample_metrics=sample_metrics,
    )
