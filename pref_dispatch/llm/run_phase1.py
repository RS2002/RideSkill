"""Phase-1 skill-repository driver (Part B, §B1): seeded directions -> QD fill.

Phase 1 builds the FROZEN skill repository the combiner (Phase 2) and repositioner
(Phase 3) later select over. It has three sub-steps; this module wires 1a and then
hands off to the existing quality-diversity loop for 1b/1c:

* **1a. Seeded directions (this module).** The researcher supplies a few skill
  *directions* in plain language (e.g. "maximise revenue per served rider",
  "minimise passenger waiting", and one FAIRNESS direction -- equalise driver
  take-home income). For EACH direction the LLM AUTHORS a fitness function grounded
  in the real episode KPIs (it can only combine the metrics the recorder emits --
  revenue / service_rate / mean_service_time / detour_total / completed / assigned /
  income_mean / income_min / income_gini / income_cv -- so it cannot invent a
  trivially-satisfiable yardstick; see
  :func:`pref_dispatch.llm.sandbox.validate_fitness`), then evolves a skill to
  maximise it via
  :func:`pref_dispatch.llm.evolve_skill_group.evolve_skill_group` -- the same
  ``(mu+lambda)`` within-scenario-GRPO search 1b/1c uses, so a researcher-required
  niche is searched exactly as hard as a self-invented one. Directed skills
  are ALWAYS kept (like the handwritten seeds): they pin researcher-required niches,
  and a fairness skill in particular must NOT be dedup-rejected for looking similar
  on efficiency KPIs -- its equity behaviour is the point.

  The fairness direction is a STANDALONE fairness-oriented SKILL (per the plan's
  decision 3): it is independent of the multiplicative wage-budget mechanism
  (``FairnessBudget``) -- the two are assumed never both active, so no double-count.

* **1b/1c. Self-invention + dedup (delegated).** After the directed skills seed the
  basis, :func:`pref_dispatch.llm.qd_basis.discover_basis` continues the QD loop:
  the LLM proposes DIFFERENTIATED new skills, each deduplicated by behavioural
  cosine similarity, until the skill cap / dry-round / max-round bound fires.

Every produced skill carries its NL objective + fitness rationale + description
(interpretability), and everything is evolved on domain-randomized scenarios with
the unchanged ``rescale="reference"`` scale-free fitness.

Live runs need ``YIBU_API_KEY`` (env only, per MEMORY ``never-write-api-key-to-repo``);
this module never touches a key -- it takes an :class:`~pref_dispatch.llm.client.LLMClient`
(the real API client, or a fake one for offline end-to-end tests).
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from pref_dispatch.llm.batch_pairing import DEFAULT_FLEET_BANDS
from pref_dispatch.llm.client import LLMClient
from pref_dispatch.llm.evolve import (
    Candidate,
    EvolutionError,
    FrozenDir,
    evolve_one_skill,
    freeze_skill,
)
from pref_dispatch.llm.evolve_skill_group import DEFAULT_FAMILY_BETA
from pref_dispatch.llm.skill_audit import (
    DEFAULT_MAX_REAUTHOR,
    evolve_skill_audited,
)
from pref_dispatch.llm.fitness_eval import (
    DEFAULT_REGIMES,
    EVAL_NUM_DRIVERS,
    EVAL_ORDER_LIMIT,
    rollout_skill_on_scenario,
)
from pref_dispatch.llm.qd_basis import (
    DEFAULT_DRY_ROUNDS,
    DEFAULT_MAX_ROUNDS,
    DEFAULT_MAX_SKILLS,
    DEFAULT_MIN_GAIN,
    DEFAULT_TAU,
    BasisSkill,
    QDResult,
    SignatureScaler,
    _raw_signature,
    cosine,
    discover_basis,
)
from pref_dispatch.llm.resume import describe_run, load_directed_leader
from pref_dispatch.scenario import Scenario, ScenarioSampler
from pref_dispatch.skills import Skill

# Researcher-set skill DIRECTIONS for step 1a. Plain-language niches the LLM turns
# into a KPI-grounded fitness + a skill. The fairness direction is standalone from
# the budget mechanism (decision 3). Override via ``directions=`` to run your own.
DEFAULT_DIRECTIONS: Sequence[str] = (
    "Maximise total realised revenue per served rider: prefer assignments that add "
    "high-fare trips, tolerating a little extra pickup distance to win them.",
    "Minimise passenger waiting and in-car time: serve the riders who can be picked "
    "up and dropped off soonest, keeping detours small.",
    "Maximise the number of distinct riders served (throughput / coverage), even at "
    "some revenue cost -- do not leave riders unmatched when a seat is free.",
    "FAIRNESS DIRECTION (driver-income equity): dispatch so that realised driver "
    "take-home income is spread EVENLY across the fleet -- lift the lowest-earning "
    "drivers and shrink the income Gini, rather than concentrating fares on a few. "
    "This is a standalone equity skill, independent of any wage-budget reweighting.",
)


@dataclass
class Phase1Result:
    """Outcome of the full Phase-1 run (directed 1a + QD 1b/1c)."""

    basis: List[BasisSkill]                 # full frozen repository (directed + QD)
    directed: List[BasisSkill] = field(default_factory=list)   # the 1a skills
    qd: Optional[QDResult] = None           # the 1b/1c result (None if skipped)
    n_directed: int = 0
    n_qd_evolved: int = 0


def _candidate_signature(
    cand: Candidate,
    scaler: SignatureScaler,
    *,
    sig_scenarios: Optional[Sequence[Scenario]],
    use_scenarios: bool,
) -> np.ndarray:
    """Normalised behavioural signature of a directed candidate.

    On the fixed ``sig_scenarios`` batch when given (v2, the shared coordinate
    frame), else on the metrics the candidate already stored per regime (v1)."""
    if use_scenarios and sig_scenarios is not None:
        metrics_list = [rollout_skill_on_scenario(cand.skill, sc) for sc in sig_scenarios]
    else:
        from pref_dispatch.llm.evolve import eval_metrics_list

        metrics_list = eval_metrics_list(cand.evaluation)
    return scaler.normalize(_raw_signature(metrics_list))


def _already_frozen(cand: Candidate, out_dir: str = FrozenDir) -> Optional[str]:
    """Path of an identical artifact already on disk, else ``None``.

    Only matters on ``resume_run``: a run that died during QD had already frozen its
    directed skills, and ``freeze_skill`` is collision-safe by RENAMING
    (``name`` -> ``name_r2``), so re-freezing would leave the repository holding the
    same program twice under two names -- inflating the basis and handing the QD
    dedup two members whose cosine is exactly 1. Identity is decided on the score
    body, not the name: a same-named artifact with different code is a genuinely
    different program and must still be frozen (under its renamed path).
    """
    meta_path = os.path.join(out_dir, f"{cand.name}.meta.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return None
    if str(meta.get("code", "")).strip() != str(cand.meta.get("code", "")).strip():
        return None
    return os.path.join(out_dir, f"{cand.name}.py")


def _basis_from_directed(
    directed_cands: Sequence[Candidate],
    *,
    sig_scenarios: Optional[Sequence[Scenario]],
    regimes: Sequence[str],
    split: str,
    seed: int,
    num_drivers: int,
    order_limit: Optional[int],
    use_scenarios: bool,
    freeze: bool,
    log: Callable[[str], None],
) -> List[BasisSkill]:
    """Freeze the directed skills and wrap them as :class:`BasisSkill` entries.

    Signatures are measured on the shared ``sig_scenarios`` frame (raw), then
    normalised by a scaler fit to the directed skills' own extremes -- a stable
    O(1) coordinate frame so the later QD dedup compares like with like. Directed
    skills are never dedup-filtered here (they are researcher-required)."""
    if not directed_cands:
        return []
    raws = []
    for cand in directed_cands:
        if use_scenarios and sig_scenarios is not None:
            metrics_list = [rollout_skill_on_scenario(cand.skill, sc) for sc in sig_scenarios]
        else:
            from pref_dispatch.llm.evolve import eval_metrics_list

            metrics_list = eval_metrics_list(cand.evaluation)
        raws.append(_raw_signature(metrics_list))
    scaler = SignatureScaler.from_signatures(raws)

    basis: List[BasisSkill] = []
    for cand, raw in zip(directed_cands, raws):
        path = None
        if freeze:
            path = _already_frozen(cand)
            if path is not None:
                log(f"[phase1 1a] {cand.name!r} is already frozen at {path}; "
                    f"not re-freezing")
            else:
                path = freeze_skill(
                    cand, regime="scenarios" if use_scenarios else "+".join(regimes)
                )
        basis.append(
            BasisSkill(
                name=cand.name,
                objective=cand.meta["objective"],
                description=cand.meta["description"],
                signature=scaler.normalize(raw),
                provenance="directed",
                candidate=cand,
                frozen_path=path,
                mechanism=str(cand.meta.get("mechanism", "") or ""),
            )
        )
        log(
            f"[phase1 1a] {'froze' if path else 'kept (not frozen)'} {cand.name!r} "
            f"signature={np.round(basis[-1].signature, 3).tolist()}"
            + (f"; {path}" if path else "")
        )
    return basis


def _discover_from_directed(
    client: LLMClient,
    env_profile: str,
    directed_cands: Sequence[Candidate],
    seeds: Sequence[Skill],
    *,
    sampler: Optional[ScenarioSampler],
    scenarios_per_round: int,
    sig_scenarios: Optional[Sequence[Scenario]],
    rescale: str,
    max_skills: int,
    tau: float,
    max_dry_rounds: int,
    max_rounds: int,
    min_gain: float,
    generations: int,
    lam: int,
    mu: int,
    crossover_rate: float,
    fresh_per_round: int,
    band_beta: float,
    bands: Sequence[Tuple[int, int]],
    workers: int,
    checkpoint_fn: Optional[Callable[[int, int, Candidate], None]],
    regimes: Sequence[str],
    split: str,
    num_drivers: int,
    order_limit: Optional[int],
    seed: int,
    temperature: float,
    max_reauthor: int,
    audit: bool,
    freeze: bool,
    patience: int,
    min_gen: int = 0,
    runoff: bool,
    log: Callable[[str], None],
) -> QDResult:
    """Run the QD self-invention loop (1b/1c) with the directed skills pre-loaded.

    :func:`discover_basis` seeds its basis from :class:`~pref_dispatch.skills.Skill`
    objects; the directed skills are already-compiled :class:`Candidate` s. We pass
    the directed skills' compiled ``.skill`` objects (which duck-type as ``Skill``:
    they carry ``.name`` and score) as extra seeds alongside any handwritten ones,
    so the self-invention loop measures novelty AGAINST the directed niches and never
    re-discovers one. Directed skills are already frozen, so we do NOT re-freeze them
    here -- only the newly self-invented skills are frozen by ``discover_basis``."""
    seed_skills: List[Skill] = list(seeds)
    for cand in directed_cands:
        sk = cand.skill
        # Attach the NL objective so the diversity cards read well in the prompt.
        try:
            setattr(sk, "objective", cand.meta["objective"])
        except Exception:  # noqa: BLE001 -- compiled skills may be read-only
            pass
        seed_skills.append(sk)

    return discover_basis(
        client, env_profile, seed_skills,
        max_skills=max_skills, tau=tau,
        max_dry_rounds=max_dry_rounds, max_rounds=max_rounds, min_gain=min_gain,
        generations=generations, lam=lam, mu=mu,
        crossover_rate=crossover_rate, fresh_per_round=fresh_per_round,
        band_beta=band_beta, bands=bands, workers=workers,
        checkpoint_fn=checkpoint_fn,
        regimes=regimes, split=split,
        num_drivers=num_drivers, order_limit=order_limit,
        seed=seed, temperature=temperature,
        max_reauthor=max_reauthor, audit=audit, freeze=freeze,
        patience=patience, min_gen=min_gen, runoff=runoff,
        sampler=sampler, scenarios_per_round=scenarios_per_round,
        sig_scenarios=sig_scenarios,
        rescale=rescale, log=log,
    )


def _evolve_directed_skill(
    client: LLMClient,
    env_profile: str,
    direction: str,
    existing_cards: Sequence[Dict],
    *,
    scenarios: Optional[Sequence[Scenario]],
    batch_fn: Optional[Callable[[int], Sequence[Scenario]]],
    rescale: str,
    generations: int,
    lam: int,
    mu: int,
    crossover_rate: float,
    fresh_per_round: int,
    band_beta: float,
    bands: Sequence[Tuple[int, int]],
    workers: int,
    checkpoint_fn: Optional[Callable[[int, Candidate], None]],
    regimes: Sequence[str],
    split: str,
    num_drivers: int,
    order_limit: Optional[int],
    seed: int,
    temperature: float,
    max_reauthor: int,
    audit: bool,
    patience: int,
    min_gen: int = 0,
    runoff: bool,
    log: Callable[[str], None],
) -> Candidate:
    """Author a KPI-grounded fitness for ``direction`` and evolve a skill to it.

    The LLM self-authors the fitness (grounded, sandbox-validated) AND the score
    body from ``objective_hint=direction``; both are then FIXED for the search, so
    a directed skill is a real answer to the researcher's direction rather than a
    drift towards whatever is easy to score. ``existing_cards`` are shown so a later
    direction does not restate an earlier skill.

    With scenarios (the trained path) this is the SAME search Phase 1b/1c runs --
    ``(mu+lambda)`` with within-scenario GRPO over a batch that rotates every
    generation. Directed and self-invented skills therefore come out of one process
    and one coordinate frame; running 1a on the old hill-climb would have made the
    researcher-required niches the weakest members of the repository. Without
    scenarios there are no columns to standardise within, so the legacy hill-climb
    still runs (the v1 regression path).

    After the search, :func:`~pref_dispatch.llm.skill_audit.evolve_skill_audited`
    shows the champion's MEASURED behaviour back to the model and asks whether it is
    the skill ``direction`` asked for. Only Phase 1 needs this: it is the only phase
    whose fitness is model-authored, so it is the only one where the search can
    faithfully maximise the wrong yardstick (two frozen skills did). A rejected
    fitness is re-authored at most ``max_reauthor`` times."""
    if scenarios:
        return evolve_skill_audited(
            client, env_profile,
            intent=direction,
            max_reauthor=max_reauthor, audit=audit,
            objective_hint=direction,
            existing_skills=list(existing_cards),
            scenarios=scenarios, batch_fn=batch_fn,
            generations=generations, mu=mu, lam=lam,
            crossover_rate=crossover_rate, fresh_per_round=fresh_per_round,
            band_beta=band_beta, bands=bands, workers=workers,
            checkpoint_fn=checkpoint_fn,
            patience=patience, min_gen=min_gen, runoff=runoff,
            rng=random.Random(seed), temperature=temperature, log=log,
        )
    return evolve_one_skill(
        client, env_profile,
        objective_hint=direction,
        existing_skills=list(existing_cards),
        reference=None,                     # a directed niche has no seed baseline
        scenarios=scenarios,
        rescale=rescale,
        generations=generations, lam=lam,
        regimes=regimes, split=split,
        num_drivers=num_drivers, order_limit=order_limit,
        seed=seed, temperature=temperature, log=log,
    )


def run_phase1(
    client: LLMClient,
    env_profile: str,
    *,
    directions: Sequence[str] = DEFAULT_DIRECTIONS,
    seeds: Sequence[Skill] = (),
    sampler: Optional[ScenarioSampler] = None,
    scenarios_per_round: int = 4,
    sig_scenarios: Optional[Sequence[Scenario]] = None,
    n_sig_scenarios: int = 3,
    rescale: str = "reference",
    max_skills: int = DEFAULT_MAX_SKILLS,
    tau: float = DEFAULT_TAU,
    max_dry_rounds: int = DEFAULT_DRY_ROUNDS,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    min_gain: float = DEFAULT_MIN_GAIN,
    run_self_invention: bool = True,
    generations: int = 3,
    lam: int = 2,
    mu: int = 3,
    crossover_rate: float = 0.35,
    fresh_per_round: int = 1,
    band_beta: float = DEFAULT_FAMILY_BETA,
    bands: Sequence[Tuple[int, int]] = DEFAULT_FLEET_BANDS,
    workers: int = 1,
    checkpoint_fn: Optional[Callable[[str, int, int, Candidate], None]] = None,
    resume_run: Optional[str] = None,
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
    log: Callable[[str], None] = print,
) -> Phase1Result:
    """Build the frozen skill repository: directed skills (1a) then QD fill (1b/1c).

    Step 1a evolves ONE skill per ``directions`` entry (each with an LLM-authored,
    KPI-grounded fitness) and keeps them ALL -- they are researcher-required niches,
    treated like handwritten seeds (no dedup rejection). Step 1b/1c then calls
    :func:`~pref_dispatch.llm.qd_basis.discover_basis` seeded with the directed
    skills (plus any handwritten ``seeds``) so the self-invention loop fills the
    REMAINING niches up to ``max_skills`` without re-discovering a directed one
    (behavioural cosine dedup at ``tau``), and then HOLDS the repository at
    ``max_skills`` by replacing its most-redundant member whenever a strictly less
    redundant proposal arrives (v3). Directed skills are protected from eviction.

    When ``sampler`` is given (the generalization path) skills are evolved on fresh
    domain-randomized scenario batches and signatures are measured on a shared fixed
    ``sig_scenarios`` batch, exactly as :func:`discover_basis` does -- so directed and
    self-invented skills live in ONE coordinate frame. Both steps then run the SAME
    ``(mu+lambda)`` within-scenario-GRPO search (``mu``/``lam``/``crossover_rate``/
    ``fresh_per_round``/``band_beta``/``bands``), and ``workers`` spreads each round's
    rollouts over processes. ``run_self_invention=False`` stops after 1a (directed
    skills only). ``freeze=False`` skips writing artifacts (offline tests).

    ``checkpoint_fn(stage, index, generation, leader)`` fires once per generation of
    every inner search, with ``stage`` either ``"directed"`` (``index`` = the
    direction) or ``"qd"`` (``index`` = the QD round). A Phase-1 run is hours long
    and a single unlucky reply can end it, so the leader has to be recoverable from
    disk without the run having finished.

    Every search in BOTH steps is followed by the post-search audit
    (:mod:`pref_dispatch.llm.skill_audit`): the champion's measured behaviour is shown
    back to the model and it decides whether that is the skill it set out to build.
    A wrong description is rewritten in place; a wrong fitness is re-authored and the
    search re-run, at most ``max_reauthor`` times (then the champion is frozen with
    ``audit.status = "unresolved"``). ``audit=False`` turns the whole check off.

    ``resume_run`` names an earlier run's checkpoint subdirectory (the ``--run-tag``
    it used). Each direction whose checkpoint shows a FINISHED search -- it reached
    generation ``generations`` -- is reused instead of re-evolved; unfinished and
    unreadable ones are searched again from scratch. See
    :mod:`pref_dispatch.llm.resume` for why a partial checkpoint is never reused and
    why the QD stage cannot be resumed.
    """
    use_scenarios = sampler is not None
    if resume_run:
        for line in describe_run(resume_run):
            log(line)
    # ONE fixed signature batch shared by directed + QD skills so cosine dedup is
    # apples-to-apples (matches discover_basis's own construction).
    if use_scenarios and sig_scenarios is None:
        sig_scenarios = sampler.sample_batch(n_sig_scenarios, base_seed=seed)
    sig_scenarios = list(sig_scenarios) if sig_scenarios is not None else None

    # --- Step 1a: one skill per researcher direction (always kept). -------- #
    directed_cands: List[Candidate] = []
    directed_cards: List[Dict] = [
        {"skill_name": s.name,
         "objective": getattr(s, "objective", f"handwritten {s.name} specialist"),
         "description": (s.__doc__ or "").strip().split("\n")[0]}
        for s in seeds
    ]
    for i, direction in enumerate(directions):
        # Fresh scenario batch per direction (variety), pinned for reproducibility,
        # then rotated every generation so a skill cannot win by fitting one hour.
        round_scenarios = None
        batch_fn = None
        if use_scenarios:
            round_scenarios = sampler.sample_batch(
                scenarios_per_round, base_seed=seed + 100 * (i + 1)
            )

            def batch_fn(gen_idx: int, _i: int = i) -> Sequence[Scenario]:
                if gen_idx == 0:
                    return round_scenarios
                return sampler.sample_batch(
                    scenarios_per_round,
                    base_seed=seed + 100 * (_i + 1) + 7 * gen_idx,
                )

        log(f"[phase1 1a] direction {i + 1}/{len(directions)}: {direction[:72]}...")
        cand = None
        if resume_run:
            cand = load_directed_leader(
                resume_run, i, generations=generations, log=log
            )
            if cand is not None and audit:
                # A resumed leader is NOT re-audited: the audit's whole input is the
                # champion's measured per-scenario record, which a checkpoint does
                # not carry. Say so, rather than letting a resumed run look audited.
                log(f"[phase1 1a] direction {i + 1} resumed from checkpoint -- the "
                    f"post-search audit is SKIPPED for it (no measured record on "
                    f"disk to judge).")
        if cand is None:
            try:
                cand = _evolve_directed_skill(
                    client, env_profile, direction, directed_cards,
                    scenarios=round_scenarios, batch_fn=batch_fn, rescale=rescale,
                    generations=generations, lam=lam, mu=mu,
                    crossover_rate=crossover_rate, fresh_per_round=fresh_per_round,
                    band_beta=band_beta, bands=bands, workers=workers,
                    checkpoint_fn=((lambda g, c, _i=i: checkpoint_fn("directed", _i, g, c))
                                   if checkpoint_fn is not None else None),
                    regimes=regimes, split=split,
                    num_drivers=num_drivers, order_limit=order_limit,
                    seed=seed + 100 * (i + 1), temperature=temperature,
                    max_reauthor=max_reauthor, audit=audit,
                    patience=patience, min_gen=min_gen, runoff=runoff, log=log,
                )
            except EvolutionError as e:
                log(f"[phase1 1a] direction {i + 1} produced no valid skill ({e}); skipped")
                continue
        directed_cands.append(cand)
        directed_cards.append({
            "skill_name": cand.name,
            "objective": cand.meta["objective"],
            "description": cand.meta["description"],
        })
        log(f"[phase1 1a] kept {cand.name!r} objective={cand.meta['objective']!r}")

    # Freeze directed skills and build their BasisSkill entries (signatures share
    # the fixed sig_scenarios frame so downstream dedup is consistent).
    directed_basis = _basis_from_directed(
        directed_cands, sig_scenarios=sig_scenarios,
        regimes=regimes, split=split, seed=seed,
        num_drivers=num_drivers, order_limit=order_limit,
        use_scenarios=use_scenarios, freeze=freeze, log=log,
    )

    if not run_self_invention:
        return Phase1Result(
            basis=list(directed_basis),
            directed=list(directed_basis),
            qd=None,
            n_directed=len(directed_basis),
            n_qd_evolved=0,
        )

    # --- Step 1b/1c: QD self-invention seeded with the directed skills. ---- #
    qd = _discover_from_directed(
        client, env_profile, directed_cands, seeds,
        sampler=sampler, scenarios_per_round=scenarios_per_round,
        sig_scenarios=sig_scenarios,
        rescale=rescale, max_skills=max_skills, tau=tau,
        max_dry_rounds=max_dry_rounds, max_rounds=max_rounds, min_gain=min_gain,
        generations=generations, lam=lam, mu=mu,
        crossover_rate=crossover_rate, fresh_per_round=fresh_per_round,
        band_beta=band_beta, bands=bands, workers=workers,
        checkpoint_fn=((lambda r, g, c: checkpoint_fn("qd", r, g, c))
                       if checkpoint_fn is not None else None),
        regimes=regimes, split=split,
        num_drivers=num_drivers, order_limit=order_limit,
        seed=seed, temperature=temperature,
        max_reauthor=max_reauthor, audit=audit, freeze=freeze,
        patience=patience, min_gen=min_gen, runoff=runoff, log=log,
    )
    return Phase1Result(
        basis=list(qd.basis),
        directed=list(directed_basis),
        qd=qd,
        n_directed=len(directed_basis),
        n_qd_evolved=qd.n_evolved,
    )
