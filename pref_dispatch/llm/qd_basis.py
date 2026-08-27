"""Phase-1 quality-diversity skill discovery (§4.7): build the frozen skill basis.

Milestone 7.4 evolved ONE skill toward a named objective. This module upgrades
Phase 1 from "evolve a few fixed objectives" to a **novelty-search / QD loop**:

* Human seeds (revenue / service / enroute) pin the KNOWN Pareto extremes so the
  basis is never worse than the handwritten one.
* The LLM then self-invents its OWN objective + fitness for each *new* niche
  (``objective_hint=None``). On the domain-randomized path each proposal is then
  searched by :func:`pref_dispatch.llm.evolve_skill_group.evolve_skill_group` --
  ``(mu+lambda)`` over variants of that ONE skill, scored group-relatively WITHIN
  each scenario. The legacy fixed-point path still uses
  :func:`pref_dispatch.llm.evolve.evolve_one_skill`.
* Every candidate gets a **behavioural signature** -- a normalised vector of the
  episode metrics it produces. A candidate whose signature is cosine-similar
  (> ``tau``) to any skill already in the basis is REDUNDANT: it is rejected and
  the redundancy is fed back into the next proposal ("you behaved almost exactly
  like skill X -- target a different trade-off").
* The loop stops at ``N`` skills or after ``R`` consecutive redundant rounds.

Design deviation (honest): §4.7 lists the signature as
``[revenue, service_rate, mean_service_time, detour_total, empty_distance_ratio,
utilisation]``. The live ``EpisodeMetrics`` recorder does NOT emit
``empty_distance_ratio`` or ``utilisation`` (its keys are revenue, service_rate,
completed, assigned, mean_service_time, detour_total, income_*). Rather than bolt
new counters onto the tested simulator, the signature uses the available keys
that empirically SEPARATE the three seeds -- ``completed`` / ``assigned`` capture
throughput and selectivity (the role the missing two would have played). Measured
seed signatures are pairwise well below ``tau`` (see ``verify`` below), so the
dedup axis is real, not cosmetic.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from pref_dispatch.llm.client import LLMClient
from pref_dispatch.llm.evolve import (
    Candidate,
    EvolutionError,
    discard_frozen_skill,
    eval_metrics_list,
    evolve_one_skill,
    freeze_skill,
)
from pref_dispatch.llm.evolve_skill_group import (
    DEFAULT_FAMILY_BETA,
    DEFAULT_FLEET_BANDS,
)
from pref_dispatch.llm.skill_audit import (
    DEFAULT_MAX_REAUTHOR,
    evolve_skill_audited,
)
from pref_dispatch.llm.fitness_eval import (
    DEFAULT_REGIMES,
    EVAL_NUM_DRIVERS,
    EVAL_ORDER_LIMIT,
    rollout_skill_metrics,
    rollout_skill_on_scenario,
)
from pref_dispatch.scenario import Scenario, ScenarioSampler
from pref_dispatch.skills import Skill

# Signature dimensions: the metrics keys that (measured) separate the seeds.
# Fairness axis is deliberately excluded (§4.7 / §5-pref): dedup is on behaviour,
# not equity, which is a downstream preference concern.
SIGNATURE_KEYS: Sequence[str] = (
    "revenue",
    "service_rate",
    "mean_service_time",
    "detour_total",
    "completed",
    "assigned",
)

DEFAULT_TAU = 0.98       # cosine-similarity redundancy threshold (§4.7)
# v3 skill cap N: the repository is FILLED to N and then held AT N by
# replacement (see ``discover_basis``). 4 researcher directions + 6 self-invented
# specialists; the 3 handwritten seeds are merged on top at load time
# (``basis.load_basis``), so inference sees ~13 skills.
DEFAULT_MAX_SKILLS = 10
DEFAULT_DRY_ROUNDS = 3   # stop after R consecutive rounds that changed nothing
DEFAULT_MAX_ROUNDS = 20  # hard cap on total proposals; give up & keep after it.
# A replacement must lower the repository's worst redundancy by at least this much;
# guards against churn from behavioural-signature noise.
DEFAULT_MIN_GAIN = 0.005
# Provenance classes that are never evicted: the handwritten seeds and the
# researcher-set directions (e.g. the fairness skill must survive to Phase 2/3).
DEFAULT_PROTECTED: Tuple[str, ...] = ("seed", "directed")
# v2 scenario evaluation defaults (domain-randomized generalization pressure).
DEFAULT_SCENARIOS_PER_ROUND = 4  # k random scenarios per evolution round.
DEFAULT_SIG_SCENARIOS = 3        # fixed batch for stable behavioural signatures.


# --------------------------------------------------------------------------- #
# Behavioural signature
# --------------------------------------------------------------------------- #
def _raw_signature(metrics_list: Sequence[Dict[str, float]]) -> np.ndarray:
    """Average each signature key over a list of episode-metrics dicts.

    Accepts the list of per-unit metrics dicts a candidate produced -- one per
    regime (v1 fixed point) OR one per random scenario (v2). In both cases the
    signature costs NO extra rollouts: it reuses the metrics the evaluation already
    stored (via :func:`pref_dispatch.llm.evolve.eval_metrics_list`).
    """
    metrics_list = list(metrics_list)
    vec = np.zeros(len(SIGNATURE_KEYS), dtype=float)
    if not metrics_list:
        return vec
    for metrics in metrics_list:
        vec += np.array([float(metrics.get(k, 0.0)) for k in SIGNATURE_KEYS])
    return vec / len(metrics_list)


@dataclass
class SignatureScaler:
    """Per-dimension scale so cosine sim is not dominated by revenue (~1e4).

    The three seeds pin the behavioural extremes, so their per-dimension max-abs
    is a stable coordinate frame: dividing by it puts every axis at O(1) and
    makes the cosine reflect *shape*, not units. Built once, reused for every
    candidate in a run.
    """

    scale: np.ndarray

    @classmethod
    def from_signatures(cls, raw_sigs: Sequence[np.ndarray]) -> "SignatureScaler":
        stacked = np.vstack(list(raw_sigs)) if raw_sigs else np.ones((1, len(SIGNATURE_KEYS)))
        scale = np.max(np.abs(stacked), axis=0)
        scale[scale == 0.0] = 1.0  # avoid divide-by-zero on a dead dimension
        return cls(scale=scale)

    def normalize(self, raw: np.ndarray) -> np.ndarray:
        return raw / self.scale


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors; degenerate (zero) vectors handled below.

    A zero signature is a real behaviour, not a measurement failure: a HOLD-OUT
    skill (one that only acts on cars another skill has already loaded) serves
    nobody when rolled ALONE, so every metric it reports is 0. Such a skill is
    legitimate -- the combiner can pick it to leave a car idle, and the matching
    layer already scores a NOOP dummy for exactly that.

    But two hold-out skills are the SAME behaviour, and plain cosine cannot say
    so (0/0). Left at 0.0 they would each read as maximally novel to the other,
    so the repository could fill with interchangeable hold-outs that dedup can
    never evict. Hence: both zero -> 1.0 (identical), exactly one zero -> 0.0
    (a hold-out genuinely is unlike any serving skill, and stays protected).
    """
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def redundancy_of(idx: int, sigs: Sequence[np.ndarray]) -> float:
    """How redundant entry ``idx`` is GIVEN the rest: its max cosine to any other.

    This is the replacement criterion for the v3 full-repository phase: a skill
    whose behaviour is nearly reproduced by some other member is the cheapest one
    to drop, because the repository loses almost no behavioural coverage. A lone
    entry (nothing to compare against) has redundancy ``-inf`` so it is never
    picked for eviction.
    """
    others = [s for j, s in enumerate(sigs) if j != idx]
    if not others:
        return float("-inf")
    return max(cosine(sigs[idx], o) for o in others)


# --------------------------------------------------------------------------- #
# Basis entries
# --------------------------------------------------------------------------- #
@dataclass
class BasisSkill:
    """One skill in the frozen basis: its card, provenance, and signature."""

    name: str
    objective: str
    description: str
    signature: np.ndarray          # normalised
    provenance: str                # "seed" | "evolved"
    candidate: Optional[Candidate] = None  # None for handwritten seeds
    frozen_path: Optional[str] = None
    # The one-line DECISION RULE the author claimed ("rank by fare per minute
    # including pickup"). Empty for handwritten seeds and for artifacts frozen
    # before the field existed. The repository is selected on behavioural
    # diversity, so telling the model what each incumbent DOES is the difference
    # between "propose something different" and "propose a different rule".
    mechanism: str = ""

    def card(self) -> Dict[str, str]:
        """Dict the prompt builders consume for diversity cards."""
        card = {
            "skill_name": self.name,
            "objective": self.objective,
            "description": self.description,
        }
        if self.mechanism:
            card["mechanism"] = self.mechanism
        return card


def _seed_signature(
    skill: Skill,
    *,
    regimes: Sequence[str],
    split: str,
    seed: int,
    num_drivers: int,
    order_limit: Optional[int],
    sig_scenarios: Optional[Sequence[Scenario]] = None,
) -> np.ndarray:
    """Roll a handwritten seed -> its raw behavioural signature.

    When ``sig_scenarios`` is given (v2), the seed is rolled on that fixed batch of
    randomized scenarios so seeds and evolved candidates share ONE coordinate frame
    for the generalization signature. Otherwise (v1) it is rolled per regime.
    """
    if sig_scenarios is not None:
        metrics_list = [rollout_skill_on_scenario(skill, sc) for sc in sig_scenarios]
        return _raw_signature(metrics_list)
    per_regime = [
        rollout_skill_metrics(
            skill, regime, split, seed=seed,
            num_drivers=num_drivers, order_limit=order_limit,
        )
        for regime in regimes
    ]
    return _raw_signature(per_regime)


# --------------------------------------------------------------------------- #
# The QD discovery loop
# --------------------------------------------------------------------------- #
@dataclass
class QDResult:
    basis: List[BasisSkill]
    n_evolved: int
    dry_rounds_hit: bool   # stopped because R consecutive rounds changed nothing
    rounds_used: int = 0   # total proposals attempted
    stop_reason: str = ""  # "max_rounds" | "dry_rounds"
    n_replaced: int = 0    # accepted-by-replacement rounds (full-repo phase)
    n_rejected: int = 0    # proposals that did not improve diversity / were dupes


def _repository_state_note(
    basis: Sequence[BasisSkill],
    sigs: Sequence[np.ndarray],
    *,
    at_cap: bool,
    protected: Sequence[str] = DEFAULT_PROTECTED,
) -> str:
    """Repository status shown to the LLM EVERY round (§4.7 v3).

    Lists each member with its current redundancy (max cosine to any other
    member), so the model sees which niches are crowded and which entries are
    already nearly duplicated. Once the repository is at its cap, it also states
    the competition rule -- a new skill is only kept if it is less redundant than
    the most-redundant incumbent, which it must therefore beat.

    Each member also shows its MECHANISM -- the decision rule it actually uses --
    where it has one. Redundancy is measured on behaviour, but the model can only
    aim away from behaviour it has been told about: an objective line says what a
    skill wants, and two skills wanting different things routinely converge on the
    same "rank by fare per minute" rule and land on top of each other. Seeds and
    pre-mechanism artifacts simply omit the line.
    """
    reds = [redundancy_of(i, sigs) for i in range(len(sigs))]
    # Only an unprotected member can actually be evicted, so the "first to be
    # dropped" marker must point at the most-redundant EVICTABLE entry -- marking a
    # protected one would tell the model to displace a skill it cannot displace.
    evictable = [i for i, b in enumerate(basis) if b.provenance not in protected]
    drop_i = max(evictable, key=lambda i: reds[i]) if evictable else None
    lines = []
    for i, (b, r) in enumerate(zip(basis, reds)):
        tag = "" if r == float("-inf") else f"redundancy {r:.3f}"
        if b.provenance in protected:
            tag += " (protected, never dropped)"
        crowd = "  <-- most redundant, first to be dropped" if i == drop_i else ""
        lines.append(f"  - {b.name} [{b.provenance}] {tag}{crowd}\n      {b.objective}")
        if b.mechanism:
            lines.append(f"      mechanism: {b.mechanism}")
    head = (
        f"The repository currently holds {len(basis)} skill(s). 'redundancy' is a "
        "skill's highest behavioural cosine to ANY other member (1.0 = its "
        "behaviour is already covered by someone else; low = it owns a unique "
        "niche); 'mechanism' is the decision rule that member actually uses:\n"
    )
    if not at_cap:
        tail = ("\nPropose a skill for a niche NONE of these occupy -- an "
                "uncovered corner of the trade-off space, reached by a decision "
                "rule none of the mechanisms above already implements.")
    else:
        tail = (
            "\nThe repository is FULL, so your proposal now COMPETES: it is kept "
            "only if it is behaviourally less redundant than the most-redundant "
            "incumbent above, which it would then replace. A near-duplicate of any "
            "listed skill will lose -- and re-implementing a listed MECHANISM under "
            "a new objective is the most common way to lose. Aim for a trade-off "
            "none of them makes."
        )
    return head + "\n".join(lines) + tail


def _lost_competition_hint(
    cand_name: str,
    cand_red: float,
    incumbent: BasisSkill,
    incumbent_red: float,
) -> str:
    """Feedback when a proposal failed to beat the most-redundant incumbent.

    Used as the next round's ``objective_hint`` so the model knows it was not
    merely 'similar' -- it lost a concrete comparison, and by how much.
    """
    return (
        f"Your last skill '{cand_name}' was REJECTED: its redundancy against the "
        f"repository was {cand_red:.3f}, which is NOT better than the weakest "
        f"incumbent '{incumbent.name}' at {incumbent_red:.3f} (objective: "
        f"{incumbent.objective}). To earn a slot you must occupy a niche that is "
        "further from EVERY existing skill than that. Pick a trade-off the "
        "repository visibly lacks -- sacrifice a metric they all protect, or "
        "prioritise a driver/order state they all ignore. Do NOT restate an "
        "existing objective in new words."
    )


def _similarity_note(
    sims: Optional[Sequence["tuple[BasisSkill, float]"]],
    tau: float,
) -> Optional[str]:
    """Always-on diversity signal for the NEXT proposal (§4 pt3, v2).

    Summarises how behaviourally close the LAST proposed skill landed to the
    existing basis -- the top few cosine similarities -- so the model sees, every
    round, which niches are already crowded and how close it is drifting to them.
    Returns ``None`` before the first candidate exists (nothing measured yet).
    """
    if not sims:
        return None
    ranked = sorted(sims, key=lambda t: t[1], reverse=True)[:3]
    lines = [
        f"  - vs '{b.name}' ({b.objective}): cosine {sim:.3f}"
        + ("  <-- TOO CLOSE" if sim > tau else "")
        for b, sim in ranked
    ]
    nearest, top = ranked[0]
    verdict = (
        f"Your last candidate was closest to '{nearest.name}' at cosine {top:.3f} "
        + (
            f"(above the redundancy threshold {tau:.2f} -- that niche is TAKEN)."
            if top > tau
            else f"(below the {tau:.2f} threshold, so it was distinct enough)."
        )
    )
    return (
        "Behavioural cosine similarity of your LAST candidate to the current basis "
        "(1.0 = identical behaviour, aim LOW):\n"
        + "\n".join(lines)
        + "\n"
        + verdict
        + " Steer the next skill toward an uncovered trade-off."
    )


def discover_basis(
    client: LLMClient,
    env_profile: str,
    seeds: Sequence[Skill],
    *,
    max_skills: int = DEFAULT_MAX_SKILLS,
    tau: float = DEFAULT_TAU,
    max_dry_rounds: int = DEFAULT_DRY_ROUNDS,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    min_gain: float = DEFAULT_MIN_GAIN,
    protect_provenance: Sequence[str] = DEFAULT_PROTECTED,
    generations: int = 3,
    mu: int = 3,
    lam: int = 2,
    crossover_rate: float = 0.35,
    fresh_per_round: int = 1,
    band_beta: float = DEFAULT_FAMILY_BETA,
    bands: Sequence[Tuple[int, int]] = DEFAULT_FLEET_BANDS,
    workers: int = 1,
    checkpoint_fn: Optional[Callable[[int, int, Candidate], None]] = None,
    regimes: Sequence[str] = DEFAULT_REGIMES,
    split: str = "train",
    num_drivers: int = EVAL_NUM_DRIVERS,
    order_limit: Optional[int] = EVAL_ORDER_LIMIT,
    seed: int = 0,
    temperature: float = 0.9,
    max_reauthor: int = DEFAULT_MAX_REAUTHOR,
    audit: bool = True,
    freeze: bool = True,
    patience: int = 0,
    min_gen: int = 0,
    runoff: bool = False,
    sampler: Optional[ScenarioSampler] = None,
    scenarios_per_round: int = DEFAULT_SCENARIOS_PER_ROUND,
    sig_scenarios: Optional[Sequence[Scenario]] = None,
    n_sig_scenarios: int = DEFAULT_SIG_SCENARIOS,
    rescale: str = "reference",
    log: Callable[[str], None] = print,
) -> QDResult:
    """Build the frozen skill basis: seed the extremes, then QD-fill new niches.

    Two evolution regimes:

    * **v2 domain-randomized (``sampler`` given, the generalization path).** Each
      round draws a fresh batch of ``scenarios_per_round`` random scenarios
      (fleet/capacity/speed/regime/preference); the candidate is scored across that
      batch (rescaled, scale-free) so it is selected for GENERALIZATION. A separate
      FIXED ``sig_scenarios`` batch (drawn once) is the common coordinate frame for
      every behavioural signature, so dedup is apples-to-apples across rounds.
    * **v1 fixed-point (``sampler`` None).** The legacy 800-car / 3-regime
      evaluation; signatures are measured over ``regimes``. Kept for regression.

    The inner search differs between the two, and deliberately so. With a sampler,
    each QD round runs :func:`~pref_dispatch.llm.evolve_skill_group.evolve_skill_group`:
    ONE skill, under the ONE objective and fitness its founder reply authored, is
    searched by ``(mu+lambda)`` with within-scenario GRPO -- every variant of that
    skill is standardised against its siblings on the SAME scene, so raw scale
    cannot decide anything and a variant only wins by beating its own family on the
    hours they all ran. Scenes rotate every generation, so a variant cannot win by
    fitting one hour. Without a sampler there are no scenario columns to standardise
    within, so the legacy hill-climb (:func:`evolve_one_skill`) still runs -- it is
    the regression path, not the one being trained.

    The sampler path additionally AUDITS every finished search
    (:func:`~pref_dispatch.llm.skill_audit.evolve_skill_audited`): a self-invented
    skill declares its own objective at generation 0 and writes its own fitness, so
    the only thing standing between "the search worked" and "the search maximised the
    wrong yardstick" is a look at what the champion measurably did. A wrong
    description is rewritten in place, a wrong fitness is re-authored and the search
    re-run at most ``max_reauthor`` times. ``audit=False`` disables it.

    Two-stage loop (v3):

    * **Fill.** While the repository is below ``max_skills``, EVERY successfully
      evolved proposal is accepted -- there is NO redundancy gate ("达到规定数量前
      不替换": before the specified count, nothing is replaced or rejected on
      redundancy grounds). Diversity is steered by the prompt, which carries the
      repository state, the measured behavioural similarity of the last proposal,
      and an explicit push toward uncovered niches.
    * **Replace.** Once the repository is FULL, exploration continues and every
      proposal COMPETES: the most-redundant incumbent (highest cosine to any other
      member, ``redundancy_of``) is evicted if and only if the newcomer's own
      redundancy against the remaining members is strictly lower (by ``min_gain``).
      So the repository always holds exactly ``max_skills`` skills and each
      replacement strictly lowers total redundancy -- even a repository that is
      already somewhat redundant keeps its full size while getting more diverse.
      Researcher-directed skills are protected from eviction when
      ``protect_provenance`` covers them (default: seeds + directed skills, so a
      required niche such as fairness can never be evicted).

    Stopping: after ``max_dry_rounds`` consecutive rounds that changed nothing
    (rejected proposal or failed evolution) -- the "found nothing new" condition --
    or at the ``max_rounds`` proposal cap. Whatever the repository holds is kept;
    ``stop_reason`` records which bound fired.

    ``checkpoint_fn(qd_round, generation, leader)`` fires after every inner
    generation's selection, so a run that dies on QD round 9 keeps rounds 1-8 AND
    round 9's best-so-far. It is only called on the sampler path -- the legacy
    hill-climb has no per-generation leader to hand out.

    Returns a :class:`QDResult`; evolved skills are frozen to
    ``pref_dispatch/evolved/skills/`` (unless ``freeze=False``, for offline tests).
    """
    use_scenarios = sampler is not None
    # Fixed signature batch: ONE coordinate frame shared by seeds and every
    # candidate, so cosine dedup compares behaviour on identical scenarios.
    if use_scenarios and sig_scenarios is None:
        sig_scenarios = sampler.sample_batch(n_sig_scenarios, base_seed=seed)
    sig_scenarios = list(sig_scenarios) if sig_scenarios is not None else None

    # --- 1. Seed the basis with the handwritten extremes. ------------------ #
    seed_raws = [
        _seed_signature(
            s, regimes=regimes, split=split, seed=seed,
            num_drivers=num_drivers, order_limit=order_limit,
            sig_scenarios=sig_scenarios,
        )
        for s in seeds
    ]
    scaler = SignatureScaler.from_signatures(seed_raws)

    basis: List[BasisSkill] = []
    for s, raw in zip(seeds, seed_raws):
        basis.append(
            BasisSkill(
                name=s.name,
                objective=getattr(s, "objective", f"handwritten {s.name} specialist"),
                description=(s.__doc__ or "").strip().split("\n")[0],
                signature=scaler.normalize(raw),
                provenance="seed",
            )
        )
        log(f"[seed] {s.name!r} signature={np.round(basis[-1].signature, 3).tolist()}")

    # --- 2. QD loop: fill to N, then keep exploring by replacement. -------- #
    n_evolved = 0
    n_replaced = 0
    n_rejected = 0
    dry = 0
    rounds = 0
    diversity_hint: Optional[str] = None
    similarity_note: Optional[str] = None  # always-on, updated after each candidate
    stop_reason = "max_rounds"
    protected = tuple(protect_provenance)

    while True:
        if dry >= max_dry_rounds:
            stop_reason = "dry_rounds"
            break
        if rounds >= max_rounds:
            stop_reason = "max_rounds"
            break
        rounds += 1
        at_cap = len(basis) >= max_skills
        # Repository state goes into EVERY prompt: the members, how crowded each
        # already is, and (at cap) the competition rule it must beat.
        repo_note = _repository_state_note(
            basis, [b.signature for b in basis], at_cap=at_cap,
            protected=protected,
        )

        # v2: fresh random batch per round (variety across rounds); the champion
        # re-eval later uses more scenarios. v1: scenarios stays None.
        round_scenarios = None
        batch_fn = None
        if use_scenarios:
            round_scenarios = sampler.sample_batch(
                scenarios_per_round, base_seed=seed + 1000 * rounds
            )

            # Scenes ROTATE inside the round too: generation g gets its own batch,
            # so a variant cannot win the inner search by fitting one hour. The
            # offsets are spread far enough apart that no (QD round, generation)
            # pair can collide with another's base seed.
            def batch_fn(gen_idx: int, _r: int = rounds) -> Sequence[Scenario]:
                if gen_idx == 0:
                    return round_scenarios
                return sampler.sample_batch(
                    scenarios_per_round,
                    base_seed=seed + 1000 * _r + 7 * gen_idx,
                )

        try:
            if use_scenarios:
                cand = evolve_skill_audited(
                    client, env_profile,
                    # No external brief here: a self-invented skill is judged against
                    # the objective its OWN generation 0 declared, held fixed across
                    # re-authors so a retry cannot redefine its way to a "match".
                    intent=None,
                    max_reauthor=max_reauthor, audit=audit,
                    objective_hint=diversity_hint,  # None => open-ended self-invention
                    existing_skills=[b.card() for b in basis],
                    similarity_note=similarity_note,  # always shown (§4 pt3)
                    repository_note=repo_note,        # v3: state + diversity push
                    scenarios=round_scenarios,
                    batch_fn=batch_fn,
                    generations=generations, mu=mu, lam=lam,
                    crossover_rate=crossover_rate,
                    fresh_per_round=fresh_per_round,
                    band_beta=band_beta, bands=bands,
                    workers=workers,
                    checkpoint_fn=(
                        (lambda g, c, _r=rounds: checkpoint_fn(_r, g, c))
                        if checkpoint_fn is not None else None
                    ),
                    patience=patience, min_gen=min_gen, runoff=runoff,
                    rng=random.Random(seed + 1000 * rounds),
                    temperature=temperature, log=log,
                )
            else:
                cand = evolve_one_skill(
                    client, env_profile,
                    objective_hint=diversity_hint,
                    existing_skills=[b.card() for b in basis],
                    reference=None,             # self-invented niche: no seed baseline
                    similarity_note=similarity_note,
                    repository_note=repo_note,
                    scenarios=None,
                    rescale=rescale,
                    generations=generations, lam=lam,
                    regimes=regimes, split=split,
                    num_drivers=num_drivers, order_limit=order_limit,
                    seed=seed, temperature=temperature, log=log,
                )
        except EvolutionError as e:
            dry += 1
            log(f"[qd round {rounds}/{max_rounds}] proposal failed to evolve ({e}); "
                f"dry {dry}/{max_dry_rounds}")
            continue

        # Signature ALWAYS measured on the fixed sig_scenarios batch (v2) so the
        # coordinate frame is stable; v1 reuses the candidate's per-regime metrics.
        if use_scenarios:
            sig_metrics = [rollout_skill_on_scenario(cand.skill, sc) for sc in sig_scenarios]
        else:
            sig_metrics = eval_metrics_list(cand.evaluation)
        raw = _raw_signature(sig_metrics)
        sig = scaler.normalize(raw)
        sims = [(b, cosine(sig, b.signature)) for b in basis]
        nearest, sim = max(sims, key=lambda t: t[1])
        # Refresh the always-on diversity signal for the NEXT round's prompt.
        similarity_note = _similarity_note(sims, tau)

        evicted: Optional[BasisSkill] = None
        if not at_cap:
            # --- Fill stage: grow to N. NO redundancy gate. ----------------- #
            # Per the user's v3 rule, the repository is simply FILLED to
            # ``max_skills`` first ("达到规定数量前不替换"): every successfully
            # evolved proposal is kept, even one behaviourally close to a member.
            # Redundancy is steered by the prompt (repository state + measured
            # similarity note + the diversity push), NOT enforced by rejection --
            # the REPLACE stage below is where redundancy is pruned. A proposal
            # that failed to evolve was already counted as a dry round above.
            pass
        else:
            # --- Replace stage: beat the most-redundant evictable member. --- #
            evictable = [i for i, b in enumerate(basis)
                         if b.provenance not in protected]
            if not evictable:
                stop_reason = "all_protected"
                log(f"[qd round {rounds}/{max_rounds}] repository full and every "
                    "member is protected from eviction; stopping.")
                break
            sigs = [b.signature for b in basis]
            worst_i = max(evictable, key=lambda i: redundancy_of(i, sigs))
            worst, worst_red = basis[worst_i], redundancy_of(worst_i, sigs)
            # The newcomer's redundancy is measured against the repository it
            # would join, i.e. with the evicted member already removed.
            kept = [s for j, s in enumerate(sigs) if j != worst_i]
            cand_red = max(cosine(sig, s) for s in kept)
            if cand_red > worst_red - min_gain:
                dry += 1
                n_rejected += 1
                diversity_hint = _lost_competition_hint(
                    cand.name, cand_red, worst, worst_red)
                log(
                    f"[qd round {rounds}/{max_rounds}] LOST {cand.name!r} "
                    f"(redundancy {cand_red:.3f} vs incumbent {worst.name!r} "
                    f"{worst_red:.3f}); dry {dry}/{max_dry_rounds}"
                )
                continue
            evicted = basis.pop(worst_i)
            n_replaced += 1
            log(
                f"[qd round {rounds}/{max_rounds}] EVICT {evicted.name!r} "
                f"(redundancy {worst_red:.3f}) for {cand.name!r} ({cand_red:.3f})"
            )
            if evicted.frozen_path:
                moved = discard_frozen_skill(evicted.frozen_path)
                log(f"    evicted artifacts -> {moved}")

        # Accepted: distinct behaviour (fill) or strictly less redundant (replace).
        dry = 0
        diversity_hint = None
        path = None
        if freeze:
            path = freeze_skill(cand, regime="scenarios" if use_scenarios else "+".join(regimes))
        basis.append(
            BasisSkill(
                name=cand.name,
                objective=cand.meta["objective"],
                description=cand.meta["description"],
                signature=sig,
                provenance="evolved",
                candidate=cand,
                frozen_path=path,
                mechanism=str(cand.meta.get("mechanism", "") or ""),
            )
        )
        n_evolved += 1
        log(
            f"[qd round {rounds}/{max_rounds}] ACCEPT {cand.name!r} "
            f"(nearest {nearest.name!r} cos {sim:.3f}) -> basis size {len(basis)}"
            + (f", replaced {evicted.name!r}" if evicted else "")
            + (f"; frozen {path}" if path else "")
        )

    log(
        f"[qd] stopped on {stop_reason} after {rounds} round(s); "
        f"basis={len(basis)}/{max_skills} ({n_evolved} accepted, "
        f"{n_replaced} by replacement, {n_rejected} rejected). "
        + (
            "Kept the repository as-is (give-up-and-keep)."
            if stop_reason in ("max_rounds", "dry_rounds", "all_protected")
            else ""
        )
    )
    return QDResult(
        basis=basis,
        n_evolved=n_evolved,
        dry_rounds_hit=(stop_reason == "dry_rounds"),
        rounds_used=rounds,
        stop_reason=stop_reason,
        n_replaced=n_replaced,
        n_rejected=n_rejected,
    )
