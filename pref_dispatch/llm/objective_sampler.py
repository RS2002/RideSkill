"""Random EPISODE-OBJECTIVE sampler for the final-version training (Part B, B2/B3).

The two-layer method's headline is *zero-retrain objective adaptation*: ONE frozen
combiner (and ONE frozen repositioner) must serve ANY episode objective it never
saw, because it READS the objective ``w`` as an input. Training that property needs
a *distribution* of objectives to optimise against -- not one fixed reward. This
module is that distribution.

Each draw is an :class:`SampledObjective` carrying everything the training loop and
the artefact provenance need:

* ``reward_function`` -- the env-shape ``(driver_id, event) -> float`` the episode
  is graded by (injected into ``build_env`` AND handed to the policy as ``w`` on
  ``phi_ep`` -- the reads-w loop, see :func:`pref_dispatch.llm.combiner_eval._event_w`).
* ``w``               -- the single-arg ``w(event) -> float`` the combiner/
  repositioner reads (a fixed-driver-id binding of ``reward_function``).
* ``spec_text``       -- the prompt-ready description of the reward (for the
  ``reward_understanding`` CoT gate).
* ``label`` / ``meta``-- short human brief + provenance the frozen artefact records.

Three objective FAMILIES, mirroring the three "forms of preference" the user asked
the trainer to draw from each round (§B2):

1. ``"raw"``      -- a randomized-coefficient :class:`DefaultRewardFunction`. This is
   the "raw fitness function" family: a concrete reward drawn straight from the
   coefficient envelope, NO LLM call, so training and offline verification are
   key-free. It spans the revenue<->service<->throughput trade-off the combiner
   must learn to read. Since the v4 diversity upgrade the envelope ALSO samples
   the empty-move / idle penalties (they were pinned at 0), so the raw family
   covers the efficiency axis too.
2. ``"weights"``  -- a metric-weight vector (revenue/service/throughput/detour/
   efficiency) turned into a reward. Key-free by default (mapped analytically
   onto reward coefficients); an LLM client, when supplied, may instead AUTHOR
   the reward from the weight dict via
   :func:`pref_dispatch.llm.evolve_reward.author_reward`.
3. ``"nl"``       -- a natural-language brief AUTHORED into a reward by the LLM
   (needs a client; the key-free path skips this family). Since v6 the briefs
   themselves are also written by the model, one fresh set per batch, with every
   brief the run has already used listed back to it (:func:`propose_briefs`) --
   a fixed nine-entry bank meant a long run kept re-grading the combiner on the
   same nine objectives.
4. ``"structural"`` -- term-DIFFERENT reward families that :class:`DefaultRewardFunction`
   can never express: ``"completion"`` (pays on drop-off, not acceptance) and
   ``"pooling"`` (per-passenger pay that rewards seat-filling). Both key-free.
   These are the families that force the combiner to read the TERMS of ``w``, not
   just the sign of a few coefficients. (A third, ``"nonlinear"``, was retired in
   v10 -- every objective this project trains and evaluates on is now linear in the
   per-step event terms.)

The mix is configurable; the default leans on the key-free ``raw``/``weights``
families so a full training + verification run needs no API key, with ``nl`` folded
in only when a client is provided (matching MEMORY ``never-write-api-key-to-repo``:
the key, when used, comes only from the client's own env var).

Every draw carries a REAL objective. v6 removed the old ``None``-reward
("objective-blind") draw: it graded nothing, and each one cost a full-hour rollout
per candidate. The "don't be worse than ignoring the objective" contract is now
held up by ranking candidates against each other on the same real objective.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from pref_dispatch.llm.reward_spec import describe_reward
from ride_gym.rewards import (
    CompletionRewardFunction,
    DefaultRewardFunction,
    NonlinearRewardFunction,
    PoolingRewardFunction,
)

RewardFn = Callable[[Dict], float]           # w(event) -> float
EnvRewardFn = Callable[[int, Dict], float]    # reward_function(driver_id, event)


# --------------------------------------------------------------------------- #
# One sampled objective.                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class SampledObjective:
    """One episode objective drawn from the training distribution.

    ``reward_function`` is the env-shape reward the episode is graded by; ``w``
    mirrors it as the single-arg surface the combiner/repositioner read, and
    ``spec_text`` is the reward description for the CoT gate. All three are
    always present since v6 dropped the ``None``-reward draw -- :attr:`is_blind`
    is kept only as a cheap assertion surface and should now always be False."""

    label: str
    family: str                                  # "raw" | "weights" | "nl" | "structural"
    reward_function: Optional[EnvRewardFn]
    w: Optional[RewardFn]
    spec_text: Optional[str] = None
    meta: Dict = field(default_factory=dict)

    @property
    def is_blind(self) -> bool:
        return self.reward_function is None


def _event_w(reward_function: Optional[EnvRewardFn]) -> Optional[RewardFn]:
    """Bind the env-shape reward into the single-arg ``w(event)`` the policy reads.

    Identical convention to :func:`pref_dispatch.llm.objective._as_event_fn` and
    :func:`pref_dispatch.llm.combiner_eval._event_w` (fixed driver id 0; the reward
    ignores identity by contract), so the objective the env grades by is byte-for-
    byte the objective the combiner/repositioner sees."""
    if reward_function is None:
        return None

    def w(event: Dict) -> float:
        return float(reward_function(0, event))

    return w


# --------------------------------------------------------------------------- #
# Coefficient envelope for the key-free "raw" family.                          #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RewardRanges:
    """Sampling envelope for a randomized :class:`DefaultRewardFunction`.

    The ranges bracket the benchmark's own coefficients (assignment 1.0,
    revenue 0.01, service 0.04, detour 0.08; empty/idle 0) so the anchor objective
    sits inside the training distribution while every other draw pulls the
    revenue<->service<->throughput balance to a different operating point. The
    assignment bonus stays dominant (throughput first) but its ratio to the other
    terms varies, which is exactly the trade-off the combiner must learn to read
    off ``w``.

    Empty/idle penalties were historically pinned at 0 here, which made the
    trained combiner blind to efficiency-style objectives (rewards that punish
    deadheading and waiting). They are now sampled ranges; each draw has a mass
    at exactly 0 (see :func:`_penalty`) so the zero-penalty operating point
    remains well represented.
    """

    assignment_bonus: Sequence[float] = (0.5, 1.5)
    revenue_coef: Sequence[float] = (0.0, 0.08)
    service_time_coef: Sequence[float] = (0.0, 0.10)
    detour_coef: Sequence[float] = (0.0, 0.12)
    empty_move_penalty: Sequence[float] = (0.0, 0.08)
    idle_penalty: Sequence[float] = (0.0, 0.05)


def _penalty(
    rng: random.Random, rng_range: Sequence[float], pin_zero: float = 0.3
) -> float:
    """Draw one penalty coefficient; ``pin_zero`` of draws are exactly 0.

    A uniform draw over ``(0, max)`` almost never lands on 0, but the zero-
    penalty point is the benchmark's own operating point, so it must keep
    probability mass in the training distribution (same rationale as
    :func:`sample_strength`)."""
    if rng.random() < pin_zero:
        return 0.0
    return round(rng.uniform(*rng_range), 4)


def sample_reward_function(
    rng: random.Random, ranges: Optional[RewardRanges] = None
) -> DefaultRewardFunction:
    """Draw one randomized-coefficient :class:`DefaultRewardFunction` (key-free)."""
    rg = ranges or RewardRanges()
    return DefaultRewardFunction(
        assignment_bonus=round(rng.uniform(*rg.assignment_bonus), 4),
        revenue_coef=round(rng.uniform(*rg.revenue_coef), 4),
        service_time_coef=round(rng.uniform(*rg.service_time_coef), 4),
        detour_coef=round(rng.uniform(*rg.detour_coef), 4),
        empty_move_penalty=_penalty(rng, rg.empty_move_penalty),
        idle_penalty=_penalty(rng, rg.idle_penalty),
    )


def _raw_objective(rng: random.Random, ranges: Optional[RewardRanges]) -> SampledObjective:
    rf = sample_reward_function(rng, ranges)
    label = (
        f"raw(a{rf.assignment_bonus:g},rev{rf.revenue_coef:g},"
        f"svc{rf.service_time_coef:g},det{rf.detour_coef:g},"
        f"emp{rf.empty_move_penalty:g},idl{rf.idle_penalty:g})"
    )
    return SampledObjective(
        label=label,
        family="raw",
        reward_function=rf,
        w=_event_w(rf),
        spec_text=describe_reward(rf),
        meta={
            "authored": False,
            "reward_name": "randomized_default_reward",
            "coefficients": {
                "assignment_bonus": rf.assignment_bonus,
                "revenue_coef": rf.revenue_coef,
                "service_time_coef": rf.service_time_coef,
                "detour_coef": rf.detour_coef,
                "empty_move_penalty": rf.empty_move_penalty,
                "idle_penalty": rf.idle_penalty,
            },
        },
    )


# --------------------------------------------------------------------------- #
# Metric-weight family.                                                        #
# --------------------------------------------------------------------------- #
# A weight vector over interpretable objectives, mapped ANALYTICALLY onto reward
# coefficients so the family is key-free (an LLM client, when given, authors from
# the weight dict instead). throughput -> assignment_bonus, revenue -> revenue_coef,
# service -> service_time penalty (negative appetite for slow fulfilment), detour ->
# detour penalty, efficiency -> empty-move + idle penalties (appetite for keeping
# every car productively occupied). The mapping's scale matches the "raw" envelope
# midpoints.
_WEIGHT_KEYS = ("throughput", "revenue", "service", "detour", "efficiency")


def sample_weight_vector(rng: random.Random) -> Dict[str, float]:
    """Draw a normalized non-negative weight over the interpretable objectives."""
    raw = {k: rng.random() for k in _WEIGHT_KEYS}
    # Keep throughput meaningfully present (a dispatcher that never assigns is
    # degenerate) by flooring its share before renormalizing.
    raw["throughput"] = max(raw["throughput"], 0.25)
    total = sum(raw.values()) or 1.0
    return {k: round(v / total, 4) for k, v in raw.items()}


def reward_from_weights(weights: Dict[str, float]) -> DefaultRewardFunction:
    """Map a metric-weight vector onto a concrete reward (key-free analytic path).

    Higher ``throughput`` weight raises the assignment bonus; ``revenue`` raises the
    fare coefficient; ``service`` and ``detour`` raise the corresponding penalties;
    ``efficiency`` raises the empty-move and idle penalties. The result is a
    :class:`DefaultRewardFunction`, so it plugs into the env and the
    ``describe_reward`` spec exactly like the ``raw`` family."""
    eff = float(weights.get("efficiency", 0.0))
    return DefaultRewardFunction(
        assignment_bonus=round(0.5 + 1.5 * float(weights.get("throughput", 0.0)), 4),
        revenue_coef=round(0.08 * float(weights.get("revenue", 0.0)), 4),
        service_time_coef=round(0.10 * float(weights.get("service", 0.0)), 4),
        detour_coef=round(0.12 * float(weights.get("detour", 0.0)), 4),
        empty_move_penalty=round(0.08 * eff, 4),
        idle_penalty=round(0.05 * eff, 4),
    )


def _weights_objective(
    rng: random.Random, client, temperature: float, log
) -> SampledObjective:
    weights = sample_weight_vector(rng)
    label = "weights(" + ",".join(f"{k[:3]}{weights[k]:g}" for k in _WEIGHT_KEYS) + ")"
    if client is not None:
        # LLM authors the reward from the weight dict (the richer path when a key is
        # available); provenance carries the CoT/code.
        from pref_dispatch.llm.evolve_reward import author_reward

        authored = author_reward(client, weights, temperature=temperature, log=log)
        return SampledObjective(
            label=label,
            family="weights",
            reward_function=authored.fn,
            w=_event_w(authored.fn),
            spec_text=authored.spec_text,
            meta=dict(authored.meta, weight_vector=weights),
        )
    rf = reward_from_weights(weights)
    return SampledObjective(
        label=label,
        family="weights",
        reward_function=rf,
        w=_event_w(rf),
        spec_text=describe_reward(rf),
        meta={"authored": False, "reward_name": "weight_mapped_reward",
              "weight_vector": weights},
    )


# --------------------------------------------------------------------------- #
# Structural families (term-DIFFERENT rewards, all key-free).                  #
# --------------------------------------------------------------------------- #
# The v4 diversity upgrade: raw/weights both live inside the SAME linear term
# structure (assignment + revenue - service - detour [- empty - idle]); only the
# coefficients move. A combiner trained on that distribution reads "how much" of
# each term w wants, but never has to adapt to a reward with DIFFERENT TERMS. The
# structural families close that gap -- completion pays on drop-off, pooling on
# seat-filling -- so the trained combiner learns to read what kind of event w
# rewards, not just the coefficient balance.
#
# v10: ``nonlinear`` (progressive per-assignment bonus) was REMOVED from this
# tuple. Every objective this project trains and evaluates on is now LINEAR in the
# per-step event terms -- a fixed price per assignment / completion / seat /
# minute -- because that is the family real dispatch platforms actually pay on.
# The generator ``_nonlinear_objective`` is kept below (retired, unreachable from
# sampling) rather than deleted, so the family can be re-enabled by putting the
# name back here if that decision is ever revisited.
_STRUCTURAL_FAMILIES: Sequence[str] = ("completion", "pooling")


def _completion_spec(rf: CompletionRewardFunction) -> str:
    return (
        "The platform grades every driver with this FIXED per-driver, per-step "
        "reward. It pays on FINISHED trips, not acceptances:\n"
        f"  + {rf.completion_bonus:.4g} * (number of orders COMPLETED -- dropped off "
        "-- this step)   [the dominant term]\n"
        f"  + {rf.assignment_bonus:.4g} * (number of orders newly ASSIGNED this step) "
        "   [small nudge]\n"
        f"  - {rf.detour_coef:.4g} * signed re-routing impact on en-route orders "
        "[detour penalty]\n\n"
        "What this reward WANTS: serve riders who will actually FINISH. A delivered "
        "order is worth several acceptances, so prefer reliable near-term fulfilment "
        "over ambitious long loops; a driver whose orders keep dropping off earns "
        "repeatedly."
    )


def _pooling_spec(rf: PoolingRewardFunction) -> str:
    return (
        "The platform grades every driver with this FIXED per-driver, per-step "
        "reward. It pays per SEAT filled:\n"
        f"  + {rf.solo_bonus:.4g} per newly assigned SOLO order (one passenger)\n"
        f"  + {rf.party_bonus:.4g} * party_size per newly assigned order carrying "
        "TWO OR MORE passengers   [the dominant term -- seat-filling]\n"
        f"  - {rf.detour_coef:.4g} * signed re-routing impact on en-route orders "
        "[detour penalty]\n\n"
        "What this reward WANTS: co-load riders. A pooled order with a second party "
        "is worth several solo acceptances, so fill spare seats even at some extra "
        "detour; solo pickups are a fallback, not the goal."
    )


def _nonlinear_spec(rf: NonlinearRewardFunction) -> str:
    return (
        "The platform grades every driver with this FIXED per-driver, per-step "
        "reward. Its terms are deliberately NONLINEAR:\n"
        f"  + {rf.base_bonus:.4g} per newly assigned order, PLUS {rf.step_bonus:.4g} "
        "extra for every additional order beyond the first (the marginal value of "
        "an extra assignment GROWS: 1st order base, 2nd +step, 3rd +2*step, ...)\n"
        f"  - {rf.service_coef:.4g} * sqrt(total end-to-end service minutes of newly "
        "assigned orders)   [sublinear: moderate waits cost little]\n"
        f"  - {rf.detour_coef:.4g} * (positive detour minutes)^2   [quadratic cliff: "
        "a little re-routing is free, a lot is ruinous]\n\n"
        "What this reward WANTS: CONCENTRATE orders on fewer cars (the progressive "
        "bonus makes a 3-order step much better than three 1-order steps), tolerate "
        "moderate waits, and never stack heavy detours."
    )


def _completion_objective(rng: random.Random) -> SampledObjective:
    rf = CompletionRewardFunction(
        completion_bonus=round(rng.uniform(1.0, 4.0), 4),
        assignment_bonus=round(rng.uniform(0.0, 0.4), 4),
        detour_coef=round(rng.uniform(0.0, 0.08), 4),
    )
    return SampledObjective(
        label=(
            f"completion(comp{rf.completion_bonus:g},asg{rf.assignment_bonus:g},"
            f"det{rf.detour_coef:g})"
        ),
        family="completion",
        reward_function=rf,
        w=_event_w(rf),
        spec_text=_completion_spec(rf),
        meta={
            "authored": False,
            "reward_name": "completion_reward",
            "coefficients": {
                "completion_bonus": rf.completion_bonus,
                "assignment_bonus": rf.assignment_bonus,
                "detour_coef": rf.detour_coef,
            },
        },
    )


def _pooling_objective(rng: random.Random) -> SampledObjective:
    rf = PoolingRewardFunction(
        solo_bonus=round(rng.uniform(0.0, 0.8), 4),
        party_bonus=round(rng.uniform(0.8, 3.0), 4),
        detour_coef=round(rng.uniform(0.0, 0.10), 4),
    )
    return SampledObjective(
        label=f"pooling(solo{rf.solo_bonus:g},party{rf.party_bonus:g},det{rf.detour_coef:g})",
        family="pooling",
        reward_function=rf,
        w=_event_w(rf),
        spec_text=_pooling_spec(rf),
        meta={
            "authored": False,
            "reward_name": "pooling_reward",
            "coefficients": {
                "solo_bonus": rf.solo_bonus,
                "party_bonus": rf.party_bonus,
                "detour_coef": rf.detour_coef,
            },
        },
    )


def _nonlinear_objective(rng: random.Random) -> SampledObjective:
    """RETIRED in v10 -- not reachable from sampling; see :data:`_STRUCTURAL_FAMILIES`.

    A progressive per-assignment bonus: the Nth order accepted in a step is worth
    more than the first. Kept as code so the family can be restored by adding
    ``"nonlinear"`` back to ``_STRUCTURAL_FAMILIES``, but nothing draws it now --
    every trained and evaluated objective is linear in the per-step event terms.
    """
    rf = NonlinearRewardFunction(
        base_bonus=round(rng.uniform(0.5, 1.5), 4),
        step_bonus=round(rng.uniform(0.2, 1.0), 4),
        service_coef=round(rng.uniform(0.0, 0.10), 4),
        detour_coef=round(rng.uniform(0.05, 0.40), 4),
    )
    return SampledObjective(
        label=(
            f"nonlinear(base{rf.base_bonus:g},step{rf.step_bonus:g},"
            f"svc{rf.service_coef:g},det{rf.detour_coef:g})"
        ),
        family="nonlinear",
        reward_function=rf,
        w=_event_w(rf),
        spec_text=_nonlinear_spec(rf),
        meta={
            "authored": False,
            "reward_name": "nonlinear_reward",
            "coefficients": {
                "base_bonus": rf.base_bonus,
                "step_bonus": rf.step_bonus,
                "service_coef": rf.service_coef,
                "detour_coef": rf.detour_coef,
            },
        },
    )


def _structural_objective(
    rng: random.Random, sub: Optional[str] = None
) -> SampledObjective:
    """Draw one term-different structural objective.

    ``sub`` pins the sub-family (``completion`` / ``pooling``); ``None`` draws it
    uniformly. Pinning is what lets :meth:`ObjectiveSampler.sample_batch`
    ROUND-ROBIN the structural families instead of leaving their counts to luck --
    a uniform draw of 3 structural slots gave the 2026-08-09 v2 run ``completion``
    1, ``pooling`` 0, which is how a whole gate family ended up trained on a single
    scene (or none at all).

    ``"nonlinear"`` is retired (v10) and no longer accepted: passing it raises,
    rather than silently drawing a family the rest of the pipeline no longer
    evaluates."""
    if sub is None:
        sub = rng.choice(_STRUCTURAL_FAMILIES)
    if sub == "completion":
        return _completion_objective(rng)
    if sub == "pooling":
        return _pooling_objective(rng)
    raise ValueError(
        f"unknown structural sub-family {sub!r}; expected one of "
        f"{tuple(_STRUCTURAL_FAMILIES)}"
    )


# --------------------------------------------------------------------------- #
# Natural-language family (LLM-authored; needs a client).                      #
# --------------------------------------------------------------------------- #
# A small bank of researcher-set NL briefs the trainer can draw from. Deliberately
# spans distinct operating points so the authored rewards are genuinely different
# objectives (not paraphrases of one). The second half is term-different: briefs
# whose natural reward lives on DIFFERENT event terms than the default
# assignment/revenue/service/detour mix (drop-offs, seat-filling, empty/idle,
# pickup proximity), so the LLM-authored family also stretches beyond the linear
# envelope.
DEFAULT_NL_BRIEFS: Sequence[str] = (
    "maximise the number of riders served, even at some revenue cost",
    "prefer long, high-fare trips and tolerate a little extra pickup distance",
    "keep passenger waiting and detours low; do not over-pool",
    "balance revenue and service, but never strand riders in low-demand areas",
    "chase revenue aggressively when demand is scarce, serve broadly when it is plentiful",
    "pay drivers for COMPLETED drop-offs, not acceptances: reward finishing trips",
    "favour pooling: a ride carrying two or more parties is worth several solo rides",
    "harshly penalise empty deadheading and idle waiting: keep every car productive",
    "reward short pickup distances above all: proximity beats detour tolerance",
)


def _nl_objective(
    rng: random.Random, client, briefs: Sequence[str], temperature: float, log
) -> SampledObjective:
    from pref_dispatch.llm.evolve_reward import author_reward

    brief = rng.choice(list(briefs))
    authored = author_reward(client, brief, temperature=temperature, log=log)
    return SampledObjective(
        label=f"nl({brief[:32]}...)",
        family="nl",
        reward_function=authored.fn,
        w=_event_w(authored.fn),
        spec_text=authored.spec_text,
        meta=dict(authored.meta, brief=brief),
    )


# v6 item 7: the nine briefs above are a fixed bank, so a long run keeps grading
# the combiner on the SAME nine objectives no matter how many rounds it survives.
# The model writes fresh ones instead, told what has already been used.
_BRIEF_MIN_WORDS = 4
_BRIEF_MAX_WORDS = 40


def _brief_ok(text: object, seen_norm: set) -> bool:
    """Accept a proposed brief: a plain English sentence, new, not code."""
    if not isinstance(text, str):
        return False
    t = text.strip()
    n_words = len(t.split())
    if n_words < _BRIEF_MIN_WORDS or n_words > _BRIEF_MAX_WORDS:
        return False
    if any(ch in t for ch in "{}();=") or "def " in t:
        return False                       # a formula / code fragment, not a brief
    return " ".join(t.lower().split()) not in seen_norm


def propose_briefs(
    client,
    n: int,
    used: Sequence[str] = (),
    *,
    temperature: float = 0.9,
    attempts: int = 2,
    log: Callable[[str], None] = print,
) -> List[str]:
    """Ask the LLM for ``n`` objective briefs that differ from ``used``.

    English only -- the brief goes down the ordinary sandbox-validated
    :func:`~pref_dispatch.llm.evolve_reward.author_reward` path afterwards, so
    nothing new is executed here.

    Returns as many ACCEPTED briefs as came back (possibly fewer than ``n``, or
    ``[]`` if every attempt failed). Callers treat ``[]`` as "keep the current
    pool" -- a bad round of proposals must not stall training.
    """
    from pref_dispatch.llm.extract import extract_json
    from pref_dispatch.llm.prompts.objective_propose import build_objective_prompt

    seen_norm = {" ".join(u.lower().split()) for u in used}
    feedback = ""
    for attempt in range(1, max(1, attempts) + 1):
        prompt = build_objective_prompt(list(used), n, repair_feedback=feedback)
        try:
            raw = client.complete(prompt["system"], prompt["user"],
                                  temperature=temperature)
            data = extract_json(raw)
            proposed = data["briefs"]
            if not isinstance(proposed, list):
                raise ValueError("'briefs' is not a list")
        except Exception as e:  # noqa: BLE001 -- proposals are best-effort
            feedback = f"attempt {attempt} failed: {type(e).__name__}: {e}"
            log(f"    [objective] brief proposal {feedback}")
            continue
        out: List[str] = []
        for b in proposed:
            if _brief_ok(b, seen_norm):
                t = str(b).strip()
                out.append(t)
                seen_norm.add(" ".join(t.lower().split()))
        if out:
            return out[:n]
        feedback = (f"attempt {attempt}: every brief was rejected (too short/long, "
                    f"contained code, or repeated one already used).")
        log(f"    [objective] {feedback}")
    return []


# --------------------------------------------------------------------------- #
# The sampler.                                                                 #
# --------------------------------------------------------------------------- #
class ObjectiveSampler:
    """Draw random episode objectives to train an objective-reading policy.

    ``family_weights`` is the categorical mix over
    ``{"raw","weights","structural","nl"}``. Families needing the LLM
    (``nl``, and ``weights`` when you want authored rewards) are only reachable
    when a ``client`` is supplied; without one the sampler stays fully key-free
    (``raw`` + analytic ``weights`` + structural), so a whole training or
    offline-verification run needs no API key.

    The default mix weights the term-different ``structural`` family heavily so
    training spans DIFFERENT reward TERMS, not just coefficient shifts (the v4
    diversity fix for the objective-adaptation failure). :meth:`sample_batch`
    stratifies by default, so a small batch cannot unluckily land without the
    structural / NL families (see :meth:`_stratified_counts`).

    v6 removed the ``blind`` family (``reward_function=None``). Every scene now
    carries a real objective: with the whole archive re-rolled on each round's
    scenes, a candidate is ranked against other CANDIDATES on the same objective,
    so the "must not be worse than ignoring w" contract no longer needs a
    w=None group member to hold it up -- and each blind pair used to cost a
    second full-hour rollout per candidate for a scene that grades nothing.

    Deterministic given the seeded :class:`random.Random`."""

    def __init__(
        self,
        *,
        client=None,
        rng: Optional[random.Random] = None,
        ranges: Optional[RewardRanges] = None,
        nl_briefs: Sequence[str] = DEFAULT_NL_BRIEFS,
        family_weights: Optional[Dict[str, float]] = None,
        temperature: float = 0.7,
        structural_fraction: Optional[float] = None,
        llm_briefs: bool = True,
        briefs_per_batch: int = 4,
        log: Callable[[str], None] = print,
    ):
        self.client = client
        self.rng = rng or random.Random(0)
        self.ranges = ranges
        self.nl_briefs = tuple(nl_briefs)
        self.temperature = temperature
        # v6 item 7: with a client, ask the model for FRESH briefs before each
        # batch instead of recycling the nine-entry bank forever. Everything the
        # run has already trained on is remembered and listed back to the model,
        # which is what stops a long run from cycling through the same handful of
        # objectives. Seeded with the bank so proposals must move past it too.
        self.llm_briefs = bool(llm_briefs) and callable(
            getattr(client, "complete", None))
        self.briefs_per_batch = int(briefs_per_batch)
        self._used_briefs: List[str] = list(self.nl_briefs)
        # Share of the batch reserved for the term-different structural families
        # (completion / pooling). Defaults to the legacy minimum.
        self.structural_fraction = (
            self.MIN_STRUCTURAL_FRACTION if structural_fraction is None
            else float(structural_fraction)
        )
        self.log = log
        # Default mix: lean on the key-free families; fold NL in only with a client.
        # Structural (term-different) carries the biggest weight -- the v4
        # objective-diversity fix.
        if family_weights is None:
            if client is None:
                family_weights = {
                    "raw": 0.35, "weights": 0.20, "structural": 0.45,
                }
            else:
                family_weights = {
                    "raw": 0.22, "weights": 0.16, "structural": 0.34, "nl": 0.28,
                }
        # Drop LLM-only families when no client is available (stay key-free), and
        # the retired blind family if a caller still passes it.
        family_weights = {
            k: v for k, v in family_weights.items()
            if k != "blind" and (k != "nl" or client is not None)
        }
        total = sum(family_weights.values()) or 1.0
        self.family_weights = {k: v / total for k, v in family_weights.items()}

    def _pick_family(self) -> str:
        r = self.rng.random()
        cum = 0.0
        for fam, wt in self.family_weights.items():
            cum += wt
            if r <= cum:
                return fam
        return next(iter(self.family_weights))

    def sample(self) -> SampledObjective:
        """Draw ONE objective from the family mix (the unstratified path)."""
        fam = self._pick_family()
        if fam == "weights":
            return _weights_objective(self.rng, self.client, self.temperature,
                                      self.log)
        if fam == "structural":
            return _structural_objective(self.rng)
        if fam == "nl":
            return _nl_objective(self.rng, self.client, self.nl_briefs,
                                 self.temperature, self.log)
        return _raw_objective(self.rng, self.ranges)

    # Stratified-batch guarantees. The B2/B3 fitness is the mean over the batch;
    # an unlucky small draw (e.g. 8 objectives landing 1 structural and 0 nl --
    # close to what happened to the first Phase-2 run) makes that mean
    # indistinguishable from always-coverage. Stratification fixes the
    # COMPOSITION of every batch, not its luck: the term-different structural
    # families must appear, and (with a client) the LLM-authored NL family must
    # be present.
    MIN_STRUCTURAL_FRACTION = 0.25

    def _stratified_counts(self, k: int) -> Dict[str, int]:
        """Deterministic per-family minimums for a batch of ``k`` objectives.

        Order of precedence when ``k`` is tiny: structural first (the term-
        different families are the ones that force reading the TERMS of ``w``),
        then NL (needs a client).

        ``structural_fraction`` (constructor) raises the structural share; at 0.5
        with ``k=12`` the three structural sub-families get 2 slots EACH via the
        round-robin in :meth:`sample_batch`, which is the minimum that lets a
        family be seen at BOTH a scarce and a large fleet once the batch is paired
        by fleet band."""
        counts: Dict[str, int] = {f: 0 for f in self.family_weights}
        counts["structural"] = min(k, max(1, math.ceil(self.structural_fraction * k)))
        rest = k - counts["structural"]
        if self.client is not None and rest > 0:
            counts["nl"] = 1
            rest -= 1
        # Any leftover slots: draw from the non-structural, non-nl families
        # (raw / weights) by their relative weights, so the batch keeps the
        # key-free linear-coefficient spread too.
        leftover = k - sum(counts.values())
        if leftover > 0:
            base = {f: w for f, w in self.family_weights.items()
                    if f not in ("structural", "nl")}
            tot = sum(base.values()) or 1.0
            for _ in range(leftover):
                r = self.rng.random()
                cum = 0.0
                for fam, wt in base.items():
                    cum += wt / tot
                    if r <= cum:
                        counts[fam] += 1
                        break
        return counts

    def sample_batch(self, k: int, stratify: bool = True) -> List[SampledObjective]:
        """Draw ``k`` objectives.

        ``stratify=True`` (default) guarantees every batch contains the
        term-different structural families and (with a client) the NL family --
        see :meth:`_stratified_counts`. The per-family order within the batch is
        then shuffled so one family cannot cluster at the front.

        With ``llm_briefs`` on, the NL pool is refreshed from the model first, so
        each batch's natural-language objectives are ones the run has not trained
        on before (v6 item 7).
        """
        if not stratify:
            self.refresh_briefs()
            objs = [self.sample() for _ in range(k)]
            self._remember_briefs(objs)
            return objs
        counts = self._stratified_counts(k)
        if counts.get("nl", 0) > 0:
            self.refresh_briefs()
        objs: List[SampledObjective] = []
        for fam, n in counts.items():
            for _i in range(n):
                if fam == "raw":
                    objs.append(_raw_objective(self.rng, self.ranges))
                elif fam == "weights":
                    objs.append(_weights_objective(self.rng, self.client,
                                                   self.temperature, self.log))
                elif fam == "structural":
                    # ROUND-ROBIN the three structural sub-families instead of
                    # drawing each uniformly: with n slots every sub-family gets
                    # floor/ceil(n/3), so none can come out 0 (see
                    # :func:`_structural_objective`).
                    objs.append(_structural_objective(
                        self.rng, _STRUCTURAL_FAMILIES[_i % len(_STRUCTURAL_FAMILIES)]))
                elif fam == "nl":
                    objs.append(_nl_objective(
                        self.rng, self.client, self.nl_briefs,
                        self.temperature, self.log))
        self.rng.shuffle(objs)
        self._remember_briefs(objs)
        return objs

    # -- LLM-proposed briefs (v6 item 7) --------------------------------- #
    def refresh_briefs(self) -> None:
        """Replace the NL pool with freshly proposed briefs (best-effort).

        Called once per batch. On any failure the current pool is kept, so a bad
        proposal round costs nothing but the call -- training never stalls on it.
        """
        if not self.llm_briefs:
            return
        fresh = propose_briefs(
            self.client, self.briefs_per_batch, self._used_briefs,
            temperature=max(self.temperature, 0.9), log=self.log,
        )
        if fresh:
            self.nl_briefs = tuple(fresh)
            self.log(f"    [objective] {len(fresh)} fresh brief(s): "
                     f"{fresh[0][:70]}...")

    def _remember_briefs(self, objs: Sequence[SampledObjective]) -> None:
        """Record which briefs this batch actually trained on, so the next
        proposal is told to move away from them too."""
        for o in objs:
            b = o.meta.get("brief")
            if isinstance(b, str) and b not in self._used_briefs:
                self._used_briefs.append(b)


# --------------------------------------------------------------------------- #
# Fairness-strength sampler (B3).                                              #
# --------------------------------------------------------------------------- #
# The Phase-3 repositioner is trained across a RANGE of fairness strengths (the
# multiplicative wage-budget aggressiveness, ``FairnessBudget(strength=...)``), so
# ONE frozen scorer best-responds across the axis rather than to a fixed strength.
# strength 0 = the identity budget (fairness off, the legacy operating point), up
# through progressively stronger wage equalisation. Drawn here so the B3 loop can
# zip a strength per (scenario, objective) pair.
#
# 2026-08-10: this range is only now a REAL spread. Both ``Preference`` and
# ``FairnessBudget`` used to clip strength to ``[0, 1]``, so every draw above 1.0
# collapsed onto exactly 1.0 -- roughly a third of each batch was the same point
# wearing different labels. Both clips are gone (floor at 0, no cap). We keep the
# upper draw at 2.0 -- above the ``[0, 1]`` band the reported experiments sweep --
# so the frozen scorer has seen stronger equalisation than it will ever be graded at.
FAIRNESS_STRENGTH_RANGE = (0.0, 2.0)


def sample_strength(
    rng: random.Random, rng_range: Sequence[float] = FAIRNESS_STRENGTH_RANGE
) -> float:
    """Draw one fairness strength; a fair share of draws pin exactly 0 (fairness
    off) so the trained scorer still covers the legacy no-fairness operating point."""
    if rng.random() < 0.25:
        return 0.0
    return round(rng.uniform(*rng_range), 3)


def sample_strengths(
    rng: random.Random, k: int, rng_range: Sequence[float] = FAIRNESS_STRENGTH_RANGE
) -> List[float]:
    return [sample_strength(rng, rng_range) for _ in range(k)]
