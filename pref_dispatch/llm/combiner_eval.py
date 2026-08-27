"""Phase-2 combiner fitness (§5.4): scalarised multi-objective over W_train.

The upper combiner is evolved to be good *on average across preferences*, which
is where the "reads the preference, generalises without retraining" property is
optimised. Fitness is the researcher-FIXED scalarisation (not self-authored --
that boundary is §5.3), reusing :func:`pref_dispatch.generalize.scalarize`:

    fitness(combiner) = mean over pref in W_train of
                        scalarize( rollout(env, combiner, pref).metrics, pref )

Two things make this an honest, comparable yardstick:

* **Fixed normalisation frame.** ``scalarize`` min-max-normalises each metric
  across a matrix of cells. If we recomputed that frame per candidate, a candidate
  could look better just by shifting the range. So the frame is built ONCE, from a
  reference matrix (the frozen-skill single-skill combiners x W_train), and reused
  for every candidate -- a stable ruler the LLM cannot game.
* **Reliability handling.** A combiner whose ``skill_scores`` frequently returns
  nothing usable falls back to an EQUAL BLEND over the whole frozen library (see
  :class:`~pref_dispatch.llm.combiner_adapter.LLMCombiner`) -- which is exactly
  the baseline the group fitness subtracts, so a program that breaks everywhere
  scores 0 without any penalty coefficient. It used to fall back to one WORKING
  single skill, which is why a ``fallback_penalty`` had to exist; it now defaults
  to 0.0.

Cost (§5.4 note): each candidate costs ``|W_train|`` rollouts. Evaluation uses the
same scarcity operating point as Phase 1 (reduced fleet, full hour) so the
preference trade-off is actually exposed and rollouts stay cheap.
"""

from __future__ import annotations

import math

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from pref_dispatch.combiner import (
    Combiner,
    EqualBlendCombiner,
    SingleSkillCombiner,
)
from pref_dispatch.evaluate import DispatchController, rollout
from pref_dispatch.generalize import METRIC_SENSE, scalarize
from pref_dispatch.llm.combiner_adapter import LLMCombiner
from pref_dispatch.llm.fitness_eval import EVAL_NUM_DRIVERS, EVAL_ORDER_LIMIT
from pref_dispatch.llm.group_fitness import (
    DEFAULT_FALLBACK_PENALTY,
    delta_advantage,
    group_advantage,
    tie_rate,
)
from pref_dispatch.matching import DEFAULT_BLEND_K
from pref_dispatch.nyc_env import make_nyc_env
from pref_dispatch.preference import Preference, sample_preferences
from pref_dispatch.scenario import Scenario, build_env
from pref_dispatch.skills import Skill

DEFAULT_REGIMES: Sequence[str] = ("offpeak", "shoulder", "peak")


def _event_w(reward_function) -> Optional[Callable[[Dict], float]]:
    """Derive the single-arg objective ``w(event) -> float`` a combiner reads from
    the env-shape ``reward_function(driver_id, event)`` it is scored under.

    The combiner/repositioner objective surface is a pure function of the per-step
    ``event`` (driver identity is not part of the objective), so we bind a fixed
    driver id 0 -- matching :func:`pref_dispatch.llm.objective._as_event_fn`. This
    is what makes the "combiner reads the SAME reward it is graded by" loop honest:
    the authored reward injected into the env is ALSO handed to the policy as ``w``
    (carried on ``phi_ep.reward_fn`` by :func:`pref_dispatch.evaluate.rollout`).
    Returns ``None`` when there is no reward (objective-blind / scalarize arm)."""
    if reward_function is None:
        return None

    def w(event: Dict) -> float:
        return float(reward_function(0, event))

    return w

# The "env_reward" objective scores the combiner by the platform's ACTUAL reward
# function -- the fleet-mean per-driver DefaultRewardFunction reward accumulated
# over the episode (``income_mean``), the SAME stream the anchor table's reward
# column sums and the SAME objective the MARL baselines optimise. Higher is better
# (sense +1). Reward magnitudes differ by orders of magnitude across scenes, so it
# is min-max normalised within each scenario's own reference frame, exactly like
# the scalarize metrics -- see :func:`scalarize_reward`.
REWARD_METRIC = "income_mean"


def scalarize_reward(metrics: Dict[str, float], frame: NormRanges) -> float:
    """Scale-free [0,1] score = the env reward (``income_mean``) min-max normalised
    against this scenario's single-skill reference frame (higher = better)."""
    lo, hi = frame[REWARD_METRIC]
    span = hi - lo
    if span <= 1e-12:
        return 0.5
    return (float(metrics[REWARD_METRIC]) - lo) / span



def make_train_prefs(n: int = 6, seed: int = 1) -> List[Preference]:
    """Sample ``W_train`` efficiency preferences (fairness off this phase)."""
    return sample_preferences(n, seed=seed, fairness_levels=(0.0,))


def _env_factory(regime: str, split: str, num_drivers: int, order_limit, seed: int):
    def factory():
        return make_nyc_env(
            seed=seed, regime=regime, split=split,
            num_drivers=num_drivers, order_limit=order_limit,
        )
    return factory


NormRanges = Dict[str, Tuple[float, float]]


def build_norm_frame(
    skills: Dict[str, Skill],
    prefs: Sequence[Preference],
    *,
    regimes: Sequence[str] = DEFAULT_REGIMES,
    split: str = "train",
    num_drivers: int = EVAL_NUM_DRIVERS,
    order_limit: Optional[int] = EVAL_ORDER_LIMIT,
    seed: int = 0,
) -> NormRanges:
    """Build the FIXED scalarisation frame from single-skill reference rollouts.

    Rolls each frozen skill (as a SingleSkillCombiner) on each regime once (the
    preference does not change a single-skill combiner's action, so one rollout
    per (skill, regime) spans the achievable metric range). The min/max of each
    scalarised metric across this reference matrix is the ruler reused for every
    candidate. Returns ``{metric: (lo, hi)}``.
    """
    vals: Dict[str, List[float]] = {m: [] for m in METRIC_SENSE}
    vals[REWARD_METRIC] = []  # env-reward frame for the "env_reward" objective
    # One neutral preference suffices: a SingleSkillCombiner ignores pref, so the
    # metrics depend only on (skill, regime).
    neutral = Preference(weights={"revenue": 0.5, "service": 0.5, "fairness": 0.0})
    for name in skills:
        comb = SingleSkillCombiner(name)
        for regime in regimes:
            env = make_nyc_env(
                seed=seed, regime=regime, split=split,
                num_drivers=num_drivers, order_limit=order_limit,
            )
            ctrl = DispatchController(comb, skills=skills)
            m = rollout(env, ctrl, neutral, seed=seed)
            for metric in vals:
                vals[metric].append(float(m[metric]))
    return {metric: (min(v), max(v)) for metric, v in vals.items()}


@dataclass
class CombinerEval:
    """Result of evaluating one combiner across W_train x regimes."""

    fitness: float                       # mean scalarised objective (penalised)
    raw_fitness: float                   # before the reliability penalty
    fallback_rate: float                 # share of decisions the program broke on
    defer_rate: float = 0.0              # share it declined to make on purpose
    per_pref: Dict[int, float] = field(default_factory=dict)     # pref index -> scalar
    per_regime: Dict[str, float] = field(default_factory=dict)   # regime -> mean scalar
    # Group-relative (B2e): objective FAMILY -> mean group ADVANTAGE. The fitness
    # is the per-objective gain over the equal-blend "no choice" baseline,
    # standardised by how much the round's programs disagree about that gain, so
    # this tells the LLM WHICH objective families it beats and which it lags --
    # the "this method is good/bad for THIS objective" feedback per family. 0.0 =
    # worth exactly as much as not choosing on that family; +1 = one whole spread
    # of the field's gains above it; NEGATIVE = choosing actively lost money.
    per_family: Dict[str, float] = field(default_factory=dict)
    # Continuity diagnostics (§5.4) -- RETIRED, always 0.0. They belonged to the
    # v1 contract where the combiner's fourth argument was a Preference and
    # "adapting" meant sliding along a revenue dial; that dial no longer exists
    # (the argument is ``w``, a reward callable). See :func:`_fleet_smoothness`.
    # The live "does the fleet move when the target moves" number is
    # ``objective_blindness`` below. Fields kept so old checkpoints still load.
    max_step_tvd: float = 0.0
    smoothness_penalty: float = 0.0
    # Objective-adaptation diagnostic (B2), REPORT-ONLY since v6: 1 minus the
    # largest total-variation distance between the fleet's argmax-skill
    # distribution under TWO DIFFERENT sampled objectives. 1.0 = the fleet picks
    # identically under every objective (it reads ``w`` and never acts on it),
    # ~0.0 = there is at least one pair of objectives under which the fleet
    # visibly switches skills. Costs no extra rollout (it is read off the
    # capture buffer of the rollouts already done) and is the paper's "does the
    # combiner actually act on the objective" number, but it no longer moves
    # fitness -- v6 grades a candidate only against other candidates on the same
    # real objective.
    objective_blindness: float = 0.0
    # 2026-08-10 DEGENERACY DIAGNOSTICS. A group-relative score is silent about
    # WHY it sits at the middle of the field: "middle of a spread field" and
    # "every candidate scored the exact same number" both print the same value
    # (0.50 under the old percentile, 0.00 under the advantage). The whole
    # Phase-2 v6 run had ``completion = 0.500`` for EVERY candidate in gens 1-5
    # and 8, which read as "average" but actually meant the family was dead -- no
    # program ever behaved differently under a completion-shaped reward, so
    # selection never saw it.
    # ``family_tie_rate[f]`` = mean over that family's pairs of the fraction of
    # OTHER CANDIDATES whose episode reward was EXACTLY equal to mine (1.0 = the
    # whole round is one behaviour; 0.0 = everyone differs).
    # ``family_skills_above[f]`` = mean number of the frozen single skills that
    # BEAT me on that family's pairs (out of len(skill_refs)) -- the absolute bar
    # a group-relative score hides.
    family_tie_rate: Dict[str, float] = field(default_factory=dict)
    family_skills_above: Dict[str, float] = field(default_factory=dict)


def _tvd(a: Dict[str, float], b: Dict[str, float], keys) -> float:
    """Total-variation distance between two fleet-mix distributions over ``keys``
    (0.0 = identical, 1.0 = disjoint support)."""
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def _fleet_smoothness(
    combiner: LLMCombiner, grid: Sequence[float]
) -> float:
    """RETIRED (v1 preference-dial measure). Always 0.0 -- do not re-enable.

    It used to be the largest total-variation jump in the fleet's argmax-skill
    distribution between adjacent points of a revenue grid, back when the
    combiner's fourth argument was a :class:`Preference` and "adapting" meant
    sliding along one revenue/service dial. That contract is gone: the fourth
    argument is now ``w``, a reward CALLABLE, so there is no revenue grid to walk
    and a ``Preference`` handed in as ``w`` is simply the wrong type -- every
    authored combiner that calls ``w(...)`` raises, gets swallowed into
    :data:`NO_PICK`, and the measure reads an identical all-NO_PICK mix at every
    grid point. It therefore returned ~0.0 regardless of the program while still
    paying for ``len(grid)`` x capture-size probe evaluations per candidate.

    The question it was asking -- does the fleet actually move when the target
    moves? -- is now :func:`_objective_blindness`, over a grid of real ``w``
    callables. Kept as a stub (rather than deleted) so the ``max_step_tvd`` /
    ``smoothness_penalty`` fields on :class:`CombinerEval` keep their meaning of
    "not measured" for old checkpoints that recorded them.
    """
    return 0.0


def _objective_grid(objectives: Sequence[object]) -> List:
    """The distinct ``w`` surfaces a combiner must differentiate across.

    One grid point per DISTINCT objective in the batch. ``w`` is the callable the
    combiner reads, so probing the same batch it is scored on is the honest "did
    it read the very objective it was graded by" check."""
    grid: List = []
    seen = set()
    for o in objectives:
        w = getattr(o, "w", None)
        key = id(w) if w is not None else "none"
        if key in seen:
            continue
        seen.add(key)
        grid.append(w)
    return grid


def blindness_from_dists(dists: Sequence[Dict[str, float]]) -> float:
    """``1 - max pairwise TVD`` over a list of fleet skill-mix distributions.

    Split out of :func:`_objective_blindness` so the parallel path can compute the
    same number from mixes measured inside a worker process (the captured driver
    sample never leaves that process -- only these few floats come back)."""
    dists = [d for d in dists if d]
    if len(dists) < 2:
        return 0.0
    keys = set()
    for d in dists:
        keys.update(d)
    max_tvd = 0.0
    for i, a in enumerate(dists):
        for b in dists[i + 1:]:
            max_tvd = max(max_tvd, _tvd(a, b, keys))
    return max(0.0, 1.0 - max_tvd)


def _objective_blindness(combiner: LLMCombiner, grid: Sequence) -> float:
    """1 - max pairwise TVD of the fleet's argmax-skill distribution across the
    distinct objectives in ``grid`` (each a ``w`` callable).

    1.0 = the fleet picks identically under EVERY objective (it reads ``w`` and
    never acts on it), 0.0 = there is at least one pair of objectives under which
    the fleet switches skill. Fewer than two grid points => nothing to
    differentiate => 0.0. Report-only since v6; it costs no extra rollout (the
    probe replays the captured driver sample) and is the paper's "does the
    combiner actually act on the objective" number.
    """
    return blindness_from_dists([combiner.fleet_pick_fractions(w) for w in grid])


def _score_cell(objective: str, metrics: Dict[str, float],
                pref: Preference, frame: NormRanges) -> float:
    """Score one rollout by the chosen objective.

    * ``"scalarize"`` -- the preference-weighted multi-objective (revenue/service/
      gini) used by v1/v2 (default; unchanged behaviour).
    * ``"env_reward"`` -- the platform's ACTUAL reward function, i.e. the env-reward
      fleet mean (``income_mean``) min-max normalised within ``frame``. This is the
      "compose FOR the current reward_func" objective.
    """
    if objective == "env_reward":
        return scalarize_reward(metrics, frame)
    if objective == "scalarize":
        return scalarize(metrics, pref, frame)
    raise ValueError(f"unknown objective {objective!r}")


def evaluate_combiner(
    combiner: Combiner,
    skills: Dict[str, Skill],
    prefs: Sequence[Preference],
    ranges: NormRanges,
    *,
    objective: str = "scalarize",
    regimes: Sequence[str] = DEFAULT_REGIMES,
    split: str = "train",
    num_drivers: int = EVAL_NUM_DRIVERS,
    order_limit: Optional[int] = EVAL_ORDER_LIMIT,
    seed: int = 0,
    fallback_penalty: float = 0.5,
    smoothness_penalty: float = 0.0,   # RETIRED, see _fleet_smoothness
    smoothness_grid: Sequence[float] = (),
) -> CombinerEval:
    """Mean scalarised objective of ``combiner`` over ``prefs`` x ``regimes``.

    ``ranges`` is the fixed frame from :func:`build_norm_frame`. When ``combiner``
    is an :class:`LLMCombiner`, its fallback rate is measured and subtracted
    (scaled by ``fallback_penalty``) so an unreliable combiner is penalised.

    ``smoothness_penalty`` is RETIRED and defaults to 0: it belonged to the v1
    contract where the combiner read a Preference and "adapting" meant sliding
    along a revenue dial (see :func:`_fleet_smoothness`). Leaving it non-zero now
    only buys probe evaluations that always return 0.
    """
    is_llm = isinstance(combiner, LLMCombiner)
    if is_llm:
        combiner.reset_telemetry()
        if smoothness_penalty:
            combiner.enable_capture(400)

    per_pref: Dict[int, float] = {}
    per_regime_accum: Dict[str, List[float]] = {r: [] for r in regimes}
    all_scalars: List[float] = []

    for pi, pref in enumerate(prefs):
        pref_scalars: List[float] = []
        for regime in regimes:
            env = make_nyc_env(
                seed=seed, regime=regime, split=split,
                num_drivers=num_drivers, order_limit=order_limit,
            )
            ctrl = DispatchController(combiner, skills=skills)
            metrics = rollout(env, ctrl, pref, seed=seed)
            s = _score_cell(objective, metrics, pref, ranges)
            pref_scalars.append(s)
            per_regime_accum[regime].append(s)
            all_scalars.append(s)
        per_pref[pi] = sum(pref_scalars) / len(pref_scalars)

    raw = sum(all_scalars) / len(all_scalars) if all_scalars else 0.0
    fb = combiner.fallback_rate if is_llm else 0.0
    max_tvd = 0.0
    if is_llm and smoothness_penalty:
        max_tvd = _fleet_smoothness(combiner, smoothness_grid)
    smooth_pen = smoothness_penalty * max_tvd
    fitness = raw - fallback_penalty * fb - smooth_pen
    return CombinerEval(
        fitness=fitness,
        raw_fitness=raw,
        fallback_rate=fb,
        per_pref=per_pref,
        per_regime={r: (sum(v) / len(v) if v else 0.0)
                    for r, v in per_regime_accum.items()},
        max_step_tvd=max_tvd,
        smoothness_penalty=smooth_pen,
    )


# =========================================================================== #
# v2: combiner fitness on domain-randomized scenarios.                        #
# =========================================================================== #
# v1 scored the combiner over W_train x three FIXED regimes at one operating
# point (800 drivers, cap 4). v2 scores it over (W_train_pref x random SCENARIO),
# where each scenario randomizes fleet/capacity/speed/regime -- so the combiner is
# selected for reading BOTH the preference AND the scene (fleet size, capacity,
# demand) before allocating. Reward magnitudes differ by orders of magnitude
# across scenarios, so we do NOT average raw metrics: scalarize() min-max
# normalizes each metric to [0,1] within a PER-SCENARIO frame (built from that
# scenario's own single-skill reference rollouts), which is already scale-free, so
# the per-scenario scalars are directly comparable and we average them honestly.


def build_scenario_norm_frame(
    skills: Dict[str, Skill],
    scenario: Scenario,
    reward_function=None,
) -> NormRanges:
    """FIXED scalarisation frame for ONE scenario from single-skill rollouts.

    Same construction as :func:`build_norm_frame` but on a concrete
    :class:`Scenario`'s env (so fleet/capacity/speed/demand match the cell the
    combiner is scored on). One rollout per frozen skill spans the achievable
    metric range on THIS scenario; the min/max is the per-scenario ruler.

    ``reward_function`` is threaded into ``build_env`` so the ``income_mean``
    (env-reward) frame's lo/hi are measured under the SAME authored reward the
    combiner is scored by -- otherwise the ``env_reward`` objective would be
    normalised against the anchor reward it is not being composed for.
    """
    vals: Dict[str, List[float]] = {m: [] for m in METRIC_SENSE}
    vals[REWARD_METRIC] = []  # env-reward frame for the "env_reward" objective
    neutral = Preference(weights={"revenue": 0.5, "service": 0.5, "fairness": 0.0})
    for name in skills:
        comb = SingleSkillCombiner(name)
        env = build_env(scenario, reward_function=reward_function)
        ctrl = DispatchController(comb, skills=skills)
        m = rollout(env, ctrl, neutral, seed=scenario.seed)
        for metric in vals:
            vals[metric].append(float(m[metric]))
    return {metric: (min(v), max(v)) for metric, v in vals.items()}


def scenario_norm_frames(
    skills: Dict[str, Skill],
    scenarios: Sequence[Scenario],
    reward_function=None,
) -> List[NormRanges]:
    """Per-scenario frames, computed once and shared across candidates in a round.

    Deterministic for a fixed scenario batch, so building these once and passing
    them to every :func:`evaluate_combiner_scenarios` call amortises the reference
    rollouts across the whole comparison round. ``reward_function`` is threaded so
    the env-reward frame is source-matched to the authored objective (§Phase-2).
    """
    return [build_scenario_norm_frame(skills, sc, reward_function=reward_function)
            for sc in scenarios]


def evaluate_combiner_scenarios(
    combiner: Combiner,
    skills: Dict[str, Skill],
    scenarios: Sequence[Scenario],
    frames: Sequence[NormRanges],
    *,
    objective: str = "scalarize",
    reward_function=None,
    fallback_penalty: float = 0.5,
    smoothness_penalty: float = 0.0,   # RETIRED, see _fleet_smoothness
    smoothness_grid: Sequence[float] = (),
) -> CombinerEval:
    """Mean scalarised objective of ``combiner`` over domain-randomized scenarios.

    Each scenario carries its OWN preference (``scenario.preference``, the
    randomized revenue/service axis) and its OWN scale-free frame (``frames[i]``
    from :func:`scenario_norm_frames`), so the combiner is scored for reading the
    preference AND the scene. The per-scenario scalars (each in [0,1]) are averaged
    -- honest across scales. The reliability penalty is applied exactly as in
    :func:`evaluate_combiner` (the smoothness one is retired; see
    :func:`_fleet_smoothness`).

    ``reward_function`` (§Phase-2) is injected into every scenario env, so under
    ``objective="env_reward"`` the ``income_mean`` each rollout reports is the
    fleet-mean cumulative value of the AUTHORED reward the combiner is composed FOR.
    """
    is_llm = isinstance(combiner, LLMCombiner)
    # The env_reward objective composes for a SINGLE fixed reward with no runtime
    # preference dial, so the step-TVD smoothness term is meaningless -- disable it.
    use_smoothness = smoothness_penalty if objective != "env_reward" else 0.0
    if is_llm:
        combiner.reset_telemetry()
        if use_smoothness:
            combiner.enable_capture(400)

    per_scenario: Dict[int, float] = {}
    all_scalars: List[float] = []
    # The combiner reads the SAME authored objective the env grades it by: the
    # reward is injected into the env AND handed to the policy as ``w`` on phi_ep,
    # so an "env_reward" candidate can actually self-derive from the objective it
    # is scored against (this is the reads-w loop the final version turns on).
    w = _event_w(reward_function)
    for i, sc in enumerate(scenarios):
        env = build_env(sc, reward_function=reward_function)
        ctrl = DispatchController(combiner, skills=skills)
        metrics = rollout(env, ctrl, sc.preference, seed=sc.seed, reward_fn=w)
        s = _score_cell(objective, metrics, sc.preference, frames[i])
        per_scenario[i] = s
        all_scalars.append(s)

    raw = sum(all_scalars) / len(all_scalars) if all_scalars else 0.0
    fb = combiner.fallback_rate if is_llm else 0.0
    max_tvd = 0.0
    if is_llm and use_smoothness:
        max_tvd = _fleet_smoothness(combiner, smoothness_grid)
    smooth_pen = use_smoothness * max_tvd
    fitness = raw - fallback_penalty * fb - smooth_pen
    return CombinerEval(
        fitness=fitness,
        raw_fitness=raw,
        fallback_rate=fb,
        per_pref=per_scenario,           # here keyed by scenario index
        per_regime={},                   # regimes are randomized per scenario
        max_step_tvd=max_tvd,
        smoothness_penalty=smooth_pen,
    )


# =========================================================================== #
# Final version (B2): combiner fitness across a DISTRIBUTION of objectives.    #
# =========================================================================== #
# The reward-conditioned arm above composes for ONE fixed reward. The final
# version's headline is zero-retrain objective adaptation: ONE frozen combiner
# must serve ANY objective it never saw, because it READS ``w``. Selecting for
# that property means scoring a candidate NOT against a single reward but against
# a BATCH of sampled objectives -- each (scenario, objective) pair injects its own
# reward into the env AND hands it to the combiner as ``w`` (the reads-w loop),
# and each pair is scored by that reward's OWN env-reward frame so the scalars are
# scale-free and averageable. The mean over the batch is the raw fitness. No pref
# slider (one objective -> one strategy) and no preference-smoothness term (the
# objective axis is spanned by the batch). The v4/v5 blindness and anti-harm
# penalties are gone in v6 -- they were proxies for "must beat ignoring the
# objective" and cost a second full rollout per pair; the group-relative fitness
# below tests the same thing directly, by ranking candidates against each other
# on the same real objective.


def build_objective_frames(
    skills: Dict[str, Skill],
    scenarios: Sequence[Scenario],
    reward_functions: Sequence[object],
) -> List[NormRanges]:
    """Per-(scenario, objective) env-reward frames, one per pair (zipped).

    Each pair's frame is built on that scenario's env UNDER that objective's
    reward (so ``income_mean``'s lo/hi are measured against the reward the
    combiner is graded by, not the anchor reward). Computed once per objective
    batch and shared across candidates in a round -- the same amortisation
    :func:`scenario_norm_frames` gives the single-reward arm.
    """
    if len(scenarios) != len(reward_functions):
        raise ValueError("scenarios and reward_functions must be the same length")
    return [
        build_scenario_norm_frame(skills, sc, reward_function=rf)
        for sc, rf in zip(scenarios, reward_functions)
    ]


def evaluate_combiner_objectives(
    combiner: Combiner,
    skills: Dict[str, Skill],
    scenarios: Sequence[Scenario],
    objectives: Sequence[object],
    frames: Sequence[NormRanges],
    *,
    fallback_penalty: float = 0.5,
) -> CombinerEval:
    """Mean env-reward score of ``combiner`` over a batch of sampled objectives.

    ``scenarios``, ``objectives`` (each a
    :class:`~pref_dispatch.llm.objective_sampler.SampledObjective`) and ``frames``
    are zipped one-to-one. For each pair the objective's ``reward_function`` is
    injected into the env AND bound onto ``phi_ep`` as ``w`` (so the combiner can
    self-derive from the very objective it is scored by), the episode is rolled,
    and the fleet-mean cumulative reward (``income_mean``) is min-max normalised in
    that pair's own frame. The per-pair scalars (each in [0,1]) are averaged --
    honest across the wildly different reward scales the batch spans.

    The reliability (fallback) penalty applies exactly as elsewhere. There is NO
    preference-smoothness term (the objective axis is spanned by the batch, not a
    runtime preference dial). ``objective_blindness`` -- how far the fleet's
    argmax-skill mix moves across the batch's own objectives -- is REPORTED (it is
    read off the capture buffer of the rollouts already done, no extra cost) but
    since v6 it does NOT move fitness; neither does any anti-harm term. Both were
    proxies for "the candidate must beat ignoring the objective"; v6 gets that
    directly by ranking candidates against each other on the same real objective
    (see the group-relative section below), which is what the Phase-2 loop uses.
    This per-pair scalarised entry point survives for the diagnostics and the
    Phase-3 arm.
    """
    if not (len(scenarios) == len(objectives) == len(frames)):
        raise ValueError("scenarios, objectives, frames must be the same length")
    is_llm = isinstance(combiner, LLMCombiner)
    if is_llm:
        combiner.reset_telemetry()
        combiner.enable_capture(400)

    per_obj: Dict[int, float] = {}
    all_scalars: List[float] = []
    for i, (sc, obj, frame) in enumerate(zip(scenarios, objectives, frames)):
        reward_function = getattr(obj, "reward_function", None)
        w = _event_w(reward_function)
        env = build_env(sc, reward_function=reward_function)
        ctrl = DispatchController(combiner, skills=skills)
        metrics = rollout(env, ctrl, sc.preference, seed=sc.seed, reward_fn=w)
        s = scalarize_reward(metrics, frame)
        per_obj[i] = s
        all_scalars.append(s)

    raw = sum(all_scalars) / len(all_scalars) if all_scalars else 0.0
    fb = combiner.fallback_rate if is_llm else 0.0
    blindness = (
        _objective_blindness(combiner, _objective_grid(objectives)) if is_llm else 0.0
    )
    return CombinerEval(
        fitness=raw - fallback_penalty * fb,
        raw_fitness=raw,
        fallback_rate=fb,
        per_pref=per_obj,                # keyed by (scenario,objective) pair index
        per_regime={},                   # objectives+scenes randomized per pair
        max_step_tvd=0.0,
        smoothness_penalty=0.0,
        objective_blindness=blindness,
    )


# =========================================================================== #
# B2e: GRPO-style GROUP-RELATIVE fitness (the user-approved training change).  #
# =========================================================================== #
# The per-pair scalarisation above (``evaluate_combiner_objectives``) normalises
# each pair against a FIXED single-skill ruler (``build_scenario_norm_frame``).
# Two problems broke it in practice:
#
#   1. **Objective INTERNAL scale.** Two objectives differing only by a 2x weight
#      have the SAME difficulty but DOUBLE the reward numbers. Per-scenario
#      min-max rescaling cannot see this: the ruler is per-scenario, not per
#      objective, so the pair's scalar is polluted by which objective it is.
#   2. **Ruler saturation.** Candidates are mixtures of skills that beat the
#      single-skill band, so nearly every candidate saturates ``scalarize_reward``
#      near 1.0 and there is no discrimination -- the arbitrary gen-0 seed never
#      loses a generation.
#
# The fix is to score candidates RELATIVELY, inside the round's OWN group, per
# (scenario, objective) pair (the GRPO estimator: collect the round's candidates
# at once, normalise WITHIN the group). Each pair's pooled group is:
#
#     {every candidate's rollout reward}             (INCLUDING the candidate itself)
#   ∪ {the frozen skills' single-skill rewards}      (computed once, deterministic;
#                                                    landmarks, OFF by default)
#
# 2026-08-12: what is standardised is no longer the raw reward but the GAIN OVER
# THE EQUAL-BLEND BASELINE -- one extra rollout per pair with every frozen skill
# weighted the same, i.e. the reward the system earns when the combiner makes no
# choice at all (:func:`_baseline_rewards`). A pair's score is
# ``delta_advantage(mine - baseline, {the same difference for every group member})``.
#
# Why the change. The centred form ``(r - group mean)/std`` says only where you
# landed among whoever happened to be alive this round, and with the single-skill
# refs OUT of the group by default (``skill_refs_in_group=False``) the group was
# candidates ONLY -- so a round in which every program was worse than not
# choosing still paid out advantages near 0, and the way to score well was to be
# the least bad of a weak field. Subtracting a FIXED, externally meaningful
# policy makes the sign absolute (positive iff choosing beat not choosing) while
# keeping every scale-free property the centred form had: the denominator is
# still the round's own spread, so a 2x objective still gives byte-identical
# advantages, and candidate and baseline are rolled on the SAME seed so the
# scene's noise cancels pairwise. See :mod:`pref_dispatch.llm.group_fitness` for
# the estimator (clipped to +/-3, dead group -> 0.0, non-finite -> -3).
#
# The fallback penalty is now 0.0 and the parameter is legacy. A program that
# breaks falls back to the equal blend, which IS the baseline, so crashing
# everywhere scores exactly 0 on its own -- there is no longer any borrowed
# credit to charge for. It used to borrow a WORKING single skill.
#
# v6 dropped the ``w=None`` group member. Every pair used to be rolled TWICE --
# once reading the objective, once ignoring it -- to hold up the "reading w must
# not be worse than ignoring it" contract. That doubled the rollout bill (a full
# real hour each) to buy one extra member out of ~20 in the group, which is why
# the 2026-08-09 champion carried a batch-mean harm of 0.05 and still lost every
# scarce-fleet completion cell of the gate. With the whole archive re-rolled on
# each round's own scenes (v6 item 6), a candidate is compared against other
# CANDIDATES on the same objective, which is a far stronger test than beating its
# own blind twin -- and the freed budget goes into more candidates and more scenes.


def _rank_score(value: float, group: Sequence[float]) -> float:
    """Deprecated alias for :func:`group_fitness.group_advantage` (2026-08-10).

    Kept only so an older caller does not break; the group it is handed must now
    INCLUDE ``value`` itself. New code should call ``group_advantage`` directly.
    """
    return group_advantage(value, group)


def _skill_reference_rewards(
    skills: Dict[str, Skill],
    scenarios: Sequence[Scenario],
    objectives: Sequence[object],
) -> List[List[float]]:
    """Per-(scenario, objective) pair, the raw env reward (``income_mean``) of
    EACH frozen skill rolled alone -- the same protocol as a candidate (same env,
    same preference, same seed, the pair's objective handed as ``w``).

    Single-skill rollouts are deterministic for a fixed scenario batch, so these
    are computed ONCE per batch and re-used as fixed group members in every round
    of the evolution -- they anchor the floor/ceiling of each pair's group the
    same way the min/max ruler did, but as concrete RANKED members instead of a
    normalisation band. ``skills`` here maps name -> Skill (its keys are the
    frozen basis)."""
    refs: List[List[float]] = []
    for sc, obj in zip(scenarios, objectives):
        reward_function = getattr(obj, "reward_function", None)
        w = _event_w(reward_function)
        env = build_env(sc, reward_function=reward_function)
        pair: List[float] = []
        for name in skills:
            ctrl = DispatchController(SingleSkillCombiner(name), skills=skills)
            m = rollout(env, ctrl, sc.preference, seed=sc.seed, reward_fn=w)
            pair.append(float(m[REWARD_METRIC]))
        refs.append(pair)
    return refs


def _baseline_rewards(
    skills: Dict[str, Skill],
    scenarios: Sequence[Scenario],
    objectives: Sequence[object],
    *,
    blend_k: int = DEFAULT_BLEND_K,
) -> List[float]:
    """Per-(scenario, objective) pair, the reward of MAKING NO CHOICE.

    One rollout per pair with :class:`~pref_dispatch.combiner.EqualBlendCombiner`:
    every frozen skill weighted the same, for every driver, every step, under
    every objective. This is the Phase-2 analogue of switching repositioning off
    in Phase 3 -- the system with the layer being trained contributed nothing --
    and :func:`_group_evals` subtracts it, which is what gives a Phase-2 fitness
    of 0.0 an absolute meaning.

    ``blend_k`` MUST match the candidates' (:meth:`CombinerCandidate.make_combiner`
    uses ``DEFAULT_BLEND_K``): the baseline is truncated by the same rule as the
    policies compared against it, so a combiner that falls back on every driver is
    byte-identical to this and scores exactly 0.

    Deterministic for a fixed batch, so a round computes it once and reuses it.
    """
    out: List[float] = []
    for sc, obj in zip(scenarios, objectives):
        reward_function = getattr(obj, "reward_function", None)
        w = _event_w(reward_function)
        env = build_env(sc, reward_function=reward_function)
        ctrl = DispatchController(
            EqualBlendCombiner(list(skills), blend_k=blend_k), skills=skills)
        m = rollout(env, ctrl, sc.preference, seed=sc.seed, reward_fn=w)
        out.append(float(m[REWARD_METRIC]))
    return out


def _delta(reward: float, baseline: float) -> float:
    """``reward - baseline``, or nan if either side is not a finite number.

    A nan reaches :func:`delta_advantage` as "this candidate crashed the whole
    rollout" and floors at ``-Z_CLIP``; it never silently contaminates the std,
    which is filtered to finite members.
    """
    try:
        d = float(reward) - float(baseline)
    except (TypeError, ValueError):
        return float("nan")
    return d if math.isfinite(d) else float("nan")


def _roll_pair_rewards(
    combiner: Combiner,
    skills: Dict[str, Skill],
    scenarios: Sequence[Scenario],
    objectives: Sequence[object],
) -> List[float]:
    """Roll ONE combiner on every (scenario, objective) pair, ONCE each, reading
    the pair's objective as ``w``.

    Returns the per-pair reward row the group accumulator appends. For an
    :class:`LLMCombiner` this also resets telemetry and enables the driver-obs
    capture BEFORE the rollouts, so the fallback rate and the blindness
    diagnostic measure THIS candidate's own run."""
    if isinstance(combiner, LLMCombiner):
        combiner.reset_telemetry()
        combiner.enable_capture(400)
    row: List[float] = []
    for sc, obj in zip(scenarios, objectives):
        reward_function = getattr(obj, "reward_function", None)
        w = _event_w(reward_function)
        env = build_env(sc, reward_function=reward_function)
        ctrl = DispatchController(combiner, skills=skills)
        metrics = rollout(env, ctrl, sc.preference, seed=sc.seed, reward_fn=w)
        row.append(float(metrics[REWARD_METRIC]))
    return row


def _group_evals(
    r_mat: Sequence[Sequence[float]],
    skill_refs: Sequence[Sequence[float]],
    objectives: Sequence[object],
    *,
    baseline: Sequence[float],
    combiners: Optional[Sequence[Combiner]] = None,
    fallback_penalty: float = 0.0,
    blindness: Optional[Sequence[float]] = None,
) -> List[CombinerEval]:
    """Score every candidate on what CHOOSING A SKILL actually bought on each pair.

    ``r_mat[i][p]`` is candidate ``i``'s reward on pair ``p``; ``baseline[p]`` is
    the reward of the equal-blend "no choice was made" policy on that same pair
    and seed (see :func:`_baseline_rewards`); ``skill_refs[p]`` are the frozen
    single skills rolled alone there (possibly empty -- they are landmarks, not
    the baseline).

    The pair score is ``delta_advantage(mine - baseline, {every candidate's mine -
    baseline} + {each single skill - baseline})``: the reward the combiner added
    over NOT CHOOSING, divided by how much the round's programs disagree about
    that quantity. Two consequences, both deliberate:

    * **The sign is absolute.** Positive means the combiner's per-driver,
      per-objective choices beat weighting every skill the same. The previous key
      was ``(mine - group mean) / group std`` over the alive programs ONLY (the
      single-skill refs default OUT of the group), so a round in which every
      program was worse than not choosing still handed out advantages near 0, and
      the way to score well was to be the least bad of a weak field. Landing in a
      weak round no longer pays.
    * **The scale is the effect's own.** Standardising against the spread of the
      DIFFERENCES puts the measurement in the units of the thing being measured.
      Candidate and baseline are rolled on the same ``sc.seed``, so each
      difference is paired and the scene's own noise cancels.

    ``fallback_penalty`` defaults to 0.0 and should stay there. A break no longer
    borrows anything to penalise: :meth:`LLMCombiner._equal_blend` returns the
    baseline policy itself, so a program that raises everywhere simply IS the
    baseline and scores 0.0 on its own. The parameter is kept so the old
    behaviour can be reproduced.

    ``combiners`` supplies the telemetry (fallback rate, capture probe) and may be
    ``None`` when scoring without live instances. ``blindness`` overrides the
    probe with numbers measured elsewhere -- the parallel path measures the fleet
    mixes inside the worker that owns the captured driver sample. ``per_family``
    gives the per-objective-family mean advantage breakdown the LLM prompt
    consumes."""
    n = len(r_mat)
    if not n:
        return []
    grid = _objective_grid(objectives)
    evals: List[CombinerEval] = []
    for i in range(n):
        raw_advs: List[float] = []
        family_accum: Dict[str, List[float]] = {}
        tie_accum: Dict[str, List[float]] = {}
        above_accum: Dict[str, List[float]] = {}
        for p, obj in enumerate(objectives):
            mine = r_mat[i][p]
            # The GRPO group is the WHOLE column (self included) plus the frozen
            # single-skill landmarks, all expressed as gains over the equal-blend
            # baseline; ``peers`` is the same set minus self, used only for the
            # degeneracy read-out below.
            peers = [r_mat[j][p] for j in range(n) if j != i]
            base = baseline[p]
            group = [_delta(r_mat[j][p], base) for j in range(n)]
            group.extend(_delta(x, base) for x in skill_refs[p])
            adv = delta_advantage(_delta(mine, base), group)
            raw_advs.append(adv)
            fam = getattr(obj, "family", "?")
            family_accum.setdefault(fam, []).append(adv)
            # Degeneracy: how much of the round is behaviourally identical to me
            # here, and how many single skills clear my bar. Both are read off
            # numbers already computed -- no extra rollout.
            tie_accum.setdefault(fam, []).append(tie_rate(mine, peers))
            above_accum.setdefault(fam, []).append(
                float(sum(1 for x in skill_refs[p] if x > mine)))
        raw = sum(raw_advs) / len(raw_advs) if raw_advs else 0.0
        fb = 0.0
        defer = 0.0
        blind = 0.0
        if combiners is not None and i < len(combiners):
            comb = combiners[i]
            if isinstance(comb, LLMCombiner):
                fb = comb.fallback_rate
                defer = comb.defer_rate
                blind = _objective_blindness(comb, grid)
        if blindness is not None and i < len(blindness):
            blind = float(blindness[i])
        evals.append(CombinerEval(
            fitness=raw - fallback_penalty * fb,
            raw_fitness=raw,
            fallback_rate=fb,
            defer_rate=defer,
            per_pref={p: a for p, a in enumerate(raw_advs)},
            per_family={f: (sum(v) / len(v)) for f, v in family_accum.items()},
            family_tie_rate={f: (sum(v) / len(v)) for f, v in tie_accum.items()},
            family_skills_above={f: (sum(v) / len(v)) for f, v in above_accum.items()},
            max_step_tvd=0.0,
            smoothness_penalty=0.0,
            objective_blindness=blind,
        ))
    return evals


def evaluate_combiner_group(
    combiners: Sequence[Combiner],
    skills: Dict[str, Skill],
    scenarios: Sequence[Scenario],
    objectives: Sequence[object],
    skill_refs: Optional[Sequence[Sequence[float]]] = None,
    *,
    baseline: Optional[Sequence[float]] = None,
    fallback_penalty: float = 0.0,
) -> List[CombinerEval]:
    """Group-relative fitness of EVERY combiner in ``combiners`` on the batch.

    One-stop entry point (used by the offline tests): rolls each candidate on
    every (scenario, objective) pair and scores them all against the pooled
    per-pair group -- see the section docstring for why the group, not a fixed
    ruler, is the fitness. ``skill_refs`` (from :func:`_skill_reference_rewards`)
    and ``baseline`` (from :func:`_baseline_rewards`) may be precomputed;
    otherwise both are rolled here (deterministic, once per call).

    The evolution loop calls the lower-level pieces (:func:`_roll_pair_rewards` +
    :func:`_group_evals`) so it can drive its own round structure.
    """
    if len(scenarios) != len(objectives):
        raise ValueError("scenarios and objectives must be the same length")
    if skill_refs is None:
        skill_refs = _skill_reference_rewards(skills, scenarios, objectives)
    if baseline is None:
        baseline = _baseline_rewards(skills, scenarios, objectives)
    r_mat = [_roll_pair_rewards(comb, skills, scenarios, objectives)
             for comb in combiners]
    return _group_evals(r_mat, skill_refs, objectives,
                        baseline=baseline,
                        combiners=list(combiners),
                        fallback_penalty=fallback_penalty)
