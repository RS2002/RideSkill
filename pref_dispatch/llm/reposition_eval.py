"""Phase-3 repositioner fitness: group-relative advantage over (scene, objective,
fairness strength) cells.

This is :mod:`pref_dispatch.llm.combiner_eval`'s B2e machinery with ONE extra axis.
Phase 2 grades a combiner on (real demand hour, sampled objective) pairs; Phase 3
grades a repositioner on the same pairs crossed with a sampled **fairness
strength** -- the aggressiveness of the wage-equalising budget the matcher applies
on top of the frozen combiner's scores. Everything else is identical, deliberately:
the same GRPO estimator (``(r - mean)/std`` inside the cell's own group, self
included, clipped at +/-3), the same ``income_mean`` reward metric, the same
fallback penalty, the same family breakdown.

**Why the strength axis is a training axis and not a deployment setting.** Turning
fairness up changes which cars are worth having where. At strength 0 the matcher is
pure efficiency, so the useful move is to put ANY idle car next to the richest
demand. At a high strength the poorest drivers' bids are multiplied up and win the
good orders regardless of position, so cruising a rich car toward that demand buys
much less than putting a poor car within reach of anything at all. A scorer trained
at one strength and deployed at another is answering the wrong question. So the
strength is sampled per cell, is visible to the scorer as
``phi_ep.fairness_strength``, and enters selection as its own ``min(...)`` term --
one frozen scorer that best-responds along the whole axis, rather than an average
that is right nowhere.

**Where the strength actually lives.** NOT on :class:`~pref_dispatch.reposition.Repositioner`
-- that class's ``strength`` is its own aggressiveness knob and always stays 1.0
here. The fairness strength rides on the :class:`~pref_dispatch.preference.Preference`
handed to :func:`~pref_dispatch.evaluate.rollout`, because that is what
``DispatchController.act`` reads to build the ``FairnessBudget``. ``Scenario.preference``
hard-codes ``fairness: 0.0``, so the sampled value is injected here
(:func:`with_fairness`) -- without that injection Phase 3 would train with fairness
permanently off and the whole axis would be a no-op.

**Anchors are not in the group.** The demand-gravity heuristic and
repositioning-off are the two externally-meaningful reference points, but they are
FIXED, so including them in every round's group would let a candidate's advantage
drift with nothing but the field around it. They run instead as a separate
fixed-batch yardstick at the end (:func:`reposition_yardstick`), mirroring Phase
2's ``skill_yardstick``. Their rewards do enter each cell's group as anchors of the
floor/ceiling, exactly as the single-skill references do in Phase 2.
"""

from __future__ import annotations

import math

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from pref_dispatch.combiner import Combiner
from pref_dispatch.evaluate import DispatchController, rollout
from pref_dispatch.llm.combiner_eval import (
    REWARD_METRIC,
    _event_w,
    _objective_grid,
    blindness_from_dists,
)
from pref_dispatch.llm.group_fitness import (
    delta_advantage,
    tie_rate,
)
from pref_dispatch.llm.reposition_adapter import GuardedScorer
from pref_dispatch.preference import Preference
from pref_dispatch.reposition import Repositioner
from pref_dispatch.scenario import Scenario, build_env
from pref_dispatch.skills import Skill

RewardFn = Callable[[Dict], float]

# Strength BANDS, the Phase-3 analogue of the objective families. Selection
# reserves an elite slot per band and charges the weakest band, so a scorer that
# is excellent with fairness off and useless with fairness on cannot win by
# averaging. The cut points are the two qualitative changes in the matcher, not
# round numbers: 0 is the budget switched OFF entirely (every driver's multiplier
# is 1), 0 < s <= 1 is the tested deployment range where the budget re-orders
# close calls, and s > 1 is the regime where the budget can override a large score
# gap outright. (The strength itself is uncapped -- see Preference.)
STRENGTH_BANDS: Sequence[str] = ("off", "mild", "strong")

# The fixed grid the report-only STRENGTH-blindness probe replays the capture on:
# one point per band (budget off / re-orders close calls / can override a large
# gap). Nothing selects on this -- it is the "does the scorer act on the fairness
# knob at all" number the paper quotes.
STRENGTH_PROBE_GRID: Sequence[float] = (0.0, 0.5, 2.0)


def strength_label(strength: float) -> str:
    """Which band a sampled fairness strength falls in."""
    s = float(strength)
    if s <= 0.0:
        return "off"
    return "mild" if s <= 1.0 else "strong"


def with_fairness(pref: Preference, strength: float) -> Preference:
    """``pref`` with its fairness term replaced by ``strength``.

    ``Scenario.preference`` always carries ``fairness: 0.0``, so this is the ONLY
    thing that turns the axis on. Revenue/service are left untouched (Preference
    renormalises them itself); the strength is floored at 0 and not capped.
    """
    weights = dict(getattr(pref, "weights", pref))
    weights["fairness"] = max(0.0, float(strength))
    return Preference(weights=weights)


@dataclass
class RepositionEval:
    """Result of evaluating one repositioner across the round's cells."""

    fitness: float                  # raw_fitness minus the reliability penalty
    raw_fitness: float              # mean group advantage over the cells
    fallback_rate: float            # share of driver decisions the program broke on
    defer_rate: float = 0.0         # share it handed back to the heuristic on purpose
    # cell index -> that cell's standardised advantage.
    per_cell: Dict[int, float] = field(default_factory=dict)
    # objective family -> mean advantage (same meaning as CombinerEval.per_family).
    per_family: Dict[str, float] = field(default_factory=dict)
    # strength band -> mean advantage. The Phase-3-only axis: this is what tells
    # the LLM "your rule works with fairness off and falls apart at strong", which
    # is the single most common failure mode of a scorer that ignores the budget.
    per_strength: Dict[str, float] = field(default_factory=dict)
    # Degeneracy read-outs, same construction as Phase 2: how much of the round
    # scored EXACTLY like me (a dead cell reads as "average" otherwise), and how
    # many of the fixed anchors beat me (the absolute bar a relative score hides).
    family_tie_rate: Dict[str, float] = field(default_factory=dict)
    family_anchors_above: Dict[str, float] = field(default_factory=dict)
    # Report-only: 1 - max pairwise TVD of the fleet's TARGET-REGION distribution
    # across the batch's distinct objectives. 1.0 = it sends cars to exactly the
    # same places no matter what it is asked to maximise.
    objective_blindness: float = 0.0
    # Report-only: the same number across STRENGTHS instead of objectives. 1.0 =
    # the scorer reads phi_ep.fairness_strength and never acts on it.
    strength_blindness: float = 0.0


# =========================================================================== #
# Rollout helpers.                                                            #
# =========================================================================== #
def _controller(
    combiner: Combiner,
    skills: Dict[str, Skill],
    scores_fn,
    *,
    reposition_strength: float = 1.0,
) -> DispatchController:
    """A controller with the FROZEN Phase-2 combiner and the candidate scorer.

    ``scores_fn=None`` with a Repositioner attached is the demand-gravity
    heuristic; passing ``repositioner=None`` (see :func:`_roll_cell_rewards`) is
    repositioning switched off entirely.
    """
    rep = Repositioner(strength=reposition_strength, scores_fn=scores_fn)
    return DispatchController(combiner, skills=skills, repositioner=rep)


def _roll_cell_rewards(
    scorer,
    combiner: Combiner,
    skills: Dict[str, Skill],
    scenarios: Sequence[Scenario],
    objectives: Sequence[object],
    strengths: Sequence[float],
    *,
    reposition_off: bool = False,
    capture: int = 400,
) -> List[float]:
    """Roll ONE repositioner over every (scene, objective, strength) cell, once.

    Returns the per-cell ``income_mean`` row the group accumulator appends. For a
    :class:`GuardedScorer` this resets telemetry and arms the region capture
    FIRST, so the fallback rate and blindness measure THIS candidate's own run.
    ``reposition_off=True`` rolls with no repositioner at all (the off anchor).
    """
    if isinstance(scorer, GuardedScorer):
        scorer.reset_telemetry()
        scorer.enable_capture(capture)
    row: List[float] = []
    for sc, obj, strength in zip(scenarios, objectives, strengths):
        reward_function = getattr(obj, "reward_function", None)
        w = _event_w(reward_function)
        env = build_env(sc, reward_function=reward_function)
        if reposition_off:
            ctrl = DispatchController(combiner, skills=skills)
        else:
            ctrl = _controller(combiner, skills, scorer)
        metrics = rollout(
            env, ctrl, with_fairness(sc.preference, strength),
            seed=sc.seed, reward_fn=w,
        )
        row.append(float(metrics[REWARD_METRIC]))
    return row


def anchor_reference_rewards(
    combiner: Combiner,
    skills: Dict[str, Skill],
    scenarios: Sequence[Scenario],
    objectives: Sequence[object],
    strengths: Sequence[float],
) -> List[List[float]]:
    """Per cell, the rewards of the two FIXED reference policies:
    the built-in demand-gravity heuristic, and repositioning switched off.

    Deterministic for a fixed batch, so a round computes them once and reuses
    them in every cell. Returns ``[[heuristic, off], ...]``, one inner list per
    cell -- the LAST entry is the BASELINE that :func:`group_evals` subtracts
    (repositioning off), the others join the group as landmarks. Order matters.
    """
    heur = _roll_cell_rewards(None, combiner, skills, scenarios, objectives, strengths)
    off = _roll_cell_rewards(None, combiner, skills, scenarios, objectives, strengths,
                             reposition_off=True)
    return [[h, o] for h, o in zip(heur, off)]


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


# =========================================================================== #
# Group-relative scoring.                                                     #
# =========================================================================== #
def _strength_blindness(
    scorer: GuardedScorer,
    w,
    grid: Sequence[float] = STRENGTH_PROBE_GRID,
) -> float:
    """1 - max pairwise TVD of the fleet's target-region mix across a fairness
    grid, at a FIXED objective.

    A real counterfactual, not a comparison of different episodes: every captured
    context is replayed with only ``phi_ep.fairness_strength`` overridden, so any
    movement in the mix is the scorer reacting to the knob and nothing else. 1.0 =
    it aims at exactly the same regions whether the budget is off or dominating.
    Report-only, like the objective version.
    """
    return blindness_from_dists(
        [scorer.fleet_region_fractions(w, fairness_strength=s) for s in grid]
    )


def group_evals(
    r_mat: Sequence[Sequence[float]],
    anchor_refs: Sequence[Sequence[float]],
    objectives: Sequence[object],
    strengths: Sequence[float],
    scorers: Optional[Sequence[object]] = None,
    *,
    fallback_penalty: float = 0.0,
    blindness: Optional[Sequence[float]] = None,
    strength_blindness: Optional[Sequence[float]] = None,
) -> List[RepositionEval]:
    """Score every candidate on what repositioning ACTUALLY BOUGHT on each cell.

    ``r_mat[i][c]`` is candidate ``i``'s reward on cell ``c``; ``anchor_refs[c]``
    is ``[demand-gravity heuristic, repositioning OFF]`` for that cell (see
    :func:`anchor_reference_rewards`).

    The cell score is ``delta_advantage(mine - off, {every candidate's mine - off}
    + {heuristic - off})``: the reward the scorer added over NOT REPOSITIONING,
    divided by how much the round's programs disagree about that quantity. Two
    consequences, both deliberate:

    * **The sign is absolute.** Positive means the scorer beat doing nothing.
      The previous key was ``(mine - group mean) / group std`` with OFF as one of
      twelve group members, so "beat doing nothing" carried about 1/12 of the
      weight and a candidate could score well by being the least bad of a weak
      round. Landing in a weak round no longer pays.
    * **The scale is the effect's own.** Repositioning moves the episode reward by
      fractions of a percent, so standardising against the spread of RAW rewards
      buried it under the between-candidate spread; standardising against the
      spread of the DIFFERENCES puts the measurement in the units of the thing
      being measured. Candidate and OFF are rolled on the same ``sc.seed``
      (:func:`_roll_cell_rewards`), so each difference is paired.

    The heuristic stays in the group as a landmark -- it keeps a sane spread in
    the denominator and preserves the "did you beat the hand-written kernel"
    reading -- but it is no longer part of the baseline.

    ``fallback_penalty`` defaults to 0.0 and should stay there. A crash no longer
    borrows anything to penalise: :meth:`GuardedScorer._stay` leaves the car
    parked, so a program that raises everywhere simply IS the OFF baseline and
    scores 0.0 on its own. The parameter is kept so the old behaviour can be
    reproduced.

    ``blindness`` / ``strength_blindness`` override the local probes with numbers
    measured inside a worker process (the parallel path), where the capture
    buffer lives. Both remain REPORT-ONLY.
    """
    n = len(r_mat)
    if not n:
        return []
    grid = _objective_grid(objectives)
    evals: List[RepositionEval] = []
    for i in range(n):
        advs: List[float] = []
        family_accum: Dict[str, List[float]] = {}
        strength_accum: Dict[str, List[float]] = {}
        tie_accum: Dict[str, List[float]] = {}
        above_accum: Dict[str, List[float]] = {}
        for c, obj in enumerate(objectives):
            mine = r_mat[i][c]
            peers = [r_mat[j][c] for j in range(n) if j != i]
            refs = list(anchor_refs[c])
            # [heuristic, off] by construction; the LAST entry is the baseline.
            off = refs[-1] if refs else 0.0
            group = [_delta(r_mat[j][c], off) for j in range(n)]
            group.extend(_delta(x, off) for x in refs[:-1])
            adv = delta_advantage(_delta(mine, off), group)
            advs.append(adv)
            fam = getattr(obj, "family", "?")
            band = strength_label(strengths[c])
            family_accum.setdefault(fam, []).append(adv)
            strength_accum.setdefault(band, []).append(adv)
            tie_accum.setdefault(fam, []).append(tie_rate(mine, peers))
            above_accum.setdefault(fam, []).append(
                float(sum(1 for x in refs if x > mine)))
        raw = sum(advs) / len(advs) if advs else 0.0

        fb = 0.0
        defer = 0.0
        blind = 0.0
        s_blind = 0.0
        if scorers is not None and i < len(scorers):
            sc_i = scorers[i]
            if isinstance(sc_i, GuardedScorer):
                fb = sc_i.fallback_rate
                defer = sc_i.defer_rate
                blind = blindness_from_dists(
                    [sc_i.fleet_region_fractions(w) for w in grid])
                s_blind = _strength_blindness(sc_i, grid[0] if grid else None)
        if blindness is not None and i < len(blindness):
            blind = float(blindness[i])
        if strength_blindness is not None and i < len(strength_blindness):
            s_blind = float(strength_blindness[i])
        evals.append(RepositionEval(
            fitness=raw - fallback_penalty * fb,
            raw_fitness=raw,
            fallback_rate=fb,
            defer_rate=defer,
            per_cell={c: a for c, a in enumerate(advs)},
            per_family={f: sum(v) / len(v) for f, v in family_accum.items()},
            per_strength={b: sum(v) / len(v) for b, v in strength_accum.items()},
            family_tie_rate={f: sum(v) / len(v) for f, v in tie_accum.items()},
            family_anchors_above={f: sum(v) / len(v) for f, v in above_accum.items()},
            objective_blindness=blind,
            strength_blindness=s_blind,
        ))
    return evals


def evaluate_repositioner_group(
    scorers: Sequence[object],
    combiner: Combiner,
    skills: Dict[str, Skill],
    scenarios: Sequence[Scenario],
    objectives: Sequence[object],
    strengths: Sequence[float],
    anchor_refs: Optional[Sequence[Sequence[float]]] = None,
    *,
    fallback_penalty: float = 0.0,
) -> List[RepositionEval]:
    """One-stop group evaluation (used by the offline tests and single-process runs).

    The evolution loop drives :func:`_roll_cell_rewards` + :func:`group_evals`
    itself so it can own its round structure, exactly like Phase 2.
    """
    if not (len(scenarios) == len(objectives) == len(strengths)):
        raise ValueError("scenarios, objectives and strengths must be the same length")
    if anchor_refs is None:
        anchor_refs = anchor_reference_rewards(
            combiner, skills, scenarios, objectives, strengths)
    r_mat = [
        _roll_cell_rewards(s, combiner, skills, scenarios, objectives, strengths)
        for s in scorers
    ]
    return group_evals(r_mat, anchor_refs, objectives, strengths,
                       scorers=list(scorers), fallback_penalty=fallback_penalty)
