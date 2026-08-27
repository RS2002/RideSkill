"""Phase-3 quick fine-tune: specialise a generalist policy to ONE concrete scenario.

Phases 1-2 freeze a GENERALIST: a domain-randomized skill basis + a combiner that
runs ZERO-SHOT across the whole scenario family. Sometimes, though, the operator
hands us ONE concrete operating point ("Tue evening peak, 1500 cars, capacity 6,
speed 40") and is willing to spend a little LLM budget to squeeze extra performance
out of the generalist policy AT THAT POINT. This module is that quick fine-tune.

The design (all three approved by the researcher):

* **Warm-start.** We do NOT re-author from scratch. Generation 0 is SEEDED directly
  from the already-frozen artifacts' code (via ``evolve_combiner(seed_code=...)`` /
  ``evolve_one_skill(seed_code=...)``, which skip the gen-0 LLM call), then a short
  improve hill-climb runs on the SINGLE concrete scenario. Minimal LLM cost.
* **Scope = combiner + only the skills it actually selects.** One capture rollout
  on the scenario (``LLMCombiner.enable_capture`` + ``fleet_pick_fractions``) tells
  us which frozen skills the combiner really uses here; we fine-tune only those, then
  the combiner on top of the improved skills. Skills the combiner never picks on this
  scene are left untouched.
* **Tagged subdir.** Fine-tuned artifacts go to
  ``pref_dispatch/evolved/finetuned/<scenario-tag>/{skills,combiners}/`` so the
  generalist basis on disk is NEVER mutated -- a fine-tune specialises a COPY.

Reward handling mirrors the base combiner: if it was frozen under an authored /
env reward (``reward_provenance``), we reconstruct that reward and fine-tune under
``objective="env_reward"`` + ``ignore_pref`` (same objective it was composed for);
otherwise the ordinary ``objective="scalarize"`` against ``scenario.preference``.

API key: read ONLY from ``YIBU_API_KEY`` / a git-ignored ``.env`` (via the shared
``make_llm_client``); this module never opens, writes, or prints the key.
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from pref_dispatch.evaluate import DispatchController, rollout
from pref_dispatch.global_stats import GlobalStats
from pref_dispatch.llm.basis import (
    EvolvedSkillsDir,
    load_basis,
    load_frozen_combiner,
)
from pref_dispatch.llm.client import make_llm_client
from pref_dispatch.llm.combiner_adapter import LLMCombiner
from pref_dispatch.llm.combiner_eval import (
    evaluate_combiner_scenarios,
    scenario_norm_frames,
)
from pref_dispatch.llm.config import LLMConfig
from pref_dispatch.llm.encode import encode_env_profile
from pref_dispatch.llm.evolve import evolve_one_skill, freeze_skill
from pref_dispatch.llm.evolve_combiner import evolve_combiner, freeze_combiner
from pref_dispatch.llm.sandbox import compile_reward
from pref_dispatch.nyc_env import make_nyc_env
from pref_dispatch.scenario import (
    Scenario,
    ScenarioRanges,
    ScenarioSampler,
    build_env,
    sample_scenario_set,
)
from pref_dispatch.skills import Skill

FinetunedRoot = os.path.join("pref_dispatch", "evolved", "finetuned")


# --------------------------------------------------------------------------- #
# Partial scenario spec: pin some axes, domain-randomize the rest.            #
# --------------------------------------------------------------------------- #
@dataclass
class ScenarioSpec:
    """A PARTIAL operating point: pin the axes the operator gave, randomize the rest.

    Phase-3 originally fine-tuned to ONE fully-concrete :class:`Scenario`. But the
    operator may hand us only a subset -- "just fleet=1000 and the default reward,
    figure out the rest" -- and expect the fine-tune to still specialise, treating
    every UNSPECIFIED axis as domain-randomized (exactly like the Phase-2 generalist
    evolution, only now with the given axes pinned).

    Any field left ``None`` is drawn per-scenario from :class:`ScenarioRanges`; any
    field set is held FIXED across the whole sampled batch. With ``n_scenarios=1``
    and every axis pinned this reduces to the original single-concrete-scenario
    fine-tune. ``split`` and ``seed`` always have concrete defaults (they are not
    randomization axes). ``pref_revenue`` is ignored when the objective is a fixed
    env reward (``ignore_pref``); it only matters for the ``scalarize`` objective.
    """

    num_drivers: Optional[int] = None
    driver_capacity: Optional[int] = None
    speed_kmh: Optional[float] = None
    regime: Optional[str] = None
    pref_revenue: Optional[float] = None
    order_limit: Optional[int] = None
    split: str = "test"
    seed: int = 0
    n_scenarios: int = 1

    @property
    def pinned(self) -> Dict[str, object]:
        """The axes the operator actually specified (for logging / the tag)."""
        out: Dict[str, object] = {}
        for k in ("num_drivers", "driver_capacity", "speed_kmh", "regime",
                  "pref_revenue", "order_limit"):
            v = getattr(self, k)
            if v is not None:
                out[k] = v
        return out

    @property
    def randomized(self) -> List[str]:
        """The axes left unspecified -> domain-randomized during fine-tuning."""
        axes = ["num_drivers", "driver_capacity", "speed_kmh", "regime", "pref_revenue"]
        return [a for a in axes if getattr(self, a) is None]

    def is_concrete(self) -> bool:
        """True iff every randomization axis is pinned (a single exact scenario)."""
        return not self.randomized and self.n_scenarios == 1

    def build_scenarios(self, ranges: Optional[ScenarioRanges] = None) -> List[Scenario]:
        """Materialise the fine-tune batch: pinned axes fixed, the rest randomized.

        Draws ``n_scenarios`` scenarios from ``ranges`` (default envelope), then
        overwrites each with the pinned axes. When everything is pinned and
        ``n_scenarios==1`` this returns exactly ``[Scenario(**pinned)]``.
        """
        rg = ranges or ScenarioRanges(order_limit=self.order_limit)
        n = max(1, int(self.n_scenarios))
        if not self.randomized and n == 1:
            # Fully concrete: no sampling, byte-identical to the legacy single point.
            return [self._concrete(seed=self.seed)]
        sampler = ScenarioSampler(ranges=rg, rng=random.Random(self.seed),
                                  split=self.split)
        batch = sampler.sample_batch(n, base_seed=self.seed)
        return [self._pin(sc) for sc in batch]

    def _concrete(self, *, seed: int) -> Scenario:
        """A Scenario with all pinned axes and defaults for the (none) unspecified."""
        rg = ScenarioRanges()
        return Scenario(
            num_drivers=self.num_drivers,
            driver_capacity=self.driver_capacity,
            speed_kmh=self.speed_kmh,
            regime=self.regime,
            split=self.split,
            order_limit=self.order_limit,
            pref_revenue=self.pref_revenue if self.pref_revenue is not None else 0.5,
            seed=seed,
        )

    def _pin(self, sc: Scenario) -> Scenario:
        """Overwrite a sampled scenario's pinned axes; keep its randomized draws."""
        if self.num_drivers is not None:
            sc.num_drivers = self.num_drivers
        if self.driver_capacity is not None:
            sc.driver_capacity = self.driver_capacity
        if self.speed_kmh is not None:
            sc.speed_kmh = self.speed_kmh
        if self.regime is not None:
            sc.regime = self.regime
        if self.pref_revenue is not None:
            sc.pref_revenue = self.pref_revenue
        if self.order_limit is not None:
            sc.order_limit = self.order_limit
        sc.split = self.split
        return sc


@dataclass
class FinetuneResult:
    """Everything the fine-tune produced, for the caller's report."""

    tag: str
    scenario: Scenario
    base_combiner_name: str
    combiner_name: str
    skills_dir: str
    combiners_dir: str
    selected_skills: List[str]
    finetuned_skills: List[str] = field(default_factory=list)
    skill_paths: Dict[str, str] = field(default_factory=dict)
    combiner_path: str = ""
    reward_name: Optional[str] = None
    ignore_pref: bool = False
    objective: str = "scalarize"
    base_fitness: float = float("nan")
    finetuned_fitness: float = float("nan")
    # Partial-spec fine-tune: the axes the operator pinned vs the randomized batch.
    pinned: Dict[str, object] = field(default_factory=dict)
    randomized_axes: List[str] = field(default_factory=list)
    n_scenarios: int = 1
    scenarios: List[Scenario] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Helpers.                                                                     #
# --------------------------------------------------------------------------- #
def scenario_tag(sc: Scenario) -> str:
    """Filesystem-safe tag for a scenario's fine-tune subdir.

    Derived from :meth:`Scenario.label` (which already encodes fleet / capacity /
    speed / simulated-clock / preference) with any non-``[A-Za-z0-9._-]`` char
    collapsed to ``_`` so it is a valid single path segment.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", sc.label()).strip("_")


def spec_tag(spec: ScenarioSpec, rep: Scenario) -> str:
    """Filesystem-safe tag for a (possibly partial) fine-tune spec.

    A fully-concrete spec tags exactly like :func:`scenario_tag` on its one
    scenario. A PARTIAL spec (some axes randomized) instead names only the pinned
    axes and marks the randomized batch, e.g. ``ft_f1000_defaultrwd_randN8`` -- so
    the subdir does not falsely claim a concrete operating point it never fixed.
    ``rep`` is the first sampled scenario, used only to complete a concrete tag.
    """
    if spec.is_concrete():
        return scenario_tag(rep)
    parts = []
    if spec.num_drivers is not None:
        parts.append(f"f{spec.num_drivers}")
    if spec.driver_capacity is not None:
        parts.append(f"c{spec.driver_capacity}")
    if spec.speed_kmh is not None:
        parts.append(f"s{spec.speed_kmh:g}")
    if spec.regime is not None:
        parts.append(spec.regime)
    if spec.pref_revenue is not None:
        parts.append(f"rev{spec.pref_revenue:g}")
    stub = "_".join(parts) if parts else "any"
    tag = f"ft_{stub}_rand{len(spec.randomized)}x{spec.n_scenarios}"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", tag).strip("_")


def _reward_from_combiner_meta(
    meta: Dict,
) -> Tuple[Optional[Callable], Optional[str], bool, Optional[str], str]:
    """Reconstruct the fine-tune objective from the base combiner's meta.

    Returns ``(reward_function, reward_name, ignore_pref, reward_spec, objective)``
    -- the SAME reward contract the base combiner was composed under, so the
    fine-tune is a local hill-climb on the identical objective (never a re-target).

    Three cases:

    * **Authored reward** -- ``reward_provenance`` carries ``authored`` + ``code``:
      recompile that sandboxed ``reward(event)`` and adapt it to the env's
      ``(driver_id, event)`` call shape (mirrors
      :func:`benchmark.run_reward_objective_sweep._reward_from_meta`). Returns the
      reconstructed fn, ``ignore_pref=True``, ``objective="env_reward"``.
    * **Env reward, no authored code** -- the combiner was frozen with
      ``ignore_pref=True`` for the env's OWN reward (e.g. ``default_reward_maximizer``
      wrapping ``DefaultRewardFunction``: ``authored=False``, no ``code``). The reward
      cannot be recompiled offline, but ``build_env(reward_function=None)`` already
      installs that same ``DefaultRewardFunction``, so we grade under
      ``objective="env_reward"`` with ``reward_function=None`` -- byte-identical to
      what the combiner was composed for. (The old fallback here silently used
      ``scalarize``, which optimises a DIFFERENT objective -- a bug for this combiner.)
    * **Legacy scalarize** -- neither of the above: ``reward_function=None``,
      ``ignore_pref=False``, ``objective="scalarize"`` (reads ``scenario.preference``).
    """
    prov = meta.get("reward_provenance") or {}
    code = prov.get("code")
    if prov.get("authored") and code:
        reward_event_fn = compile_reward(code)

        def _env_reward(_driver_id, event, _fn=reward_event_fn):
            return float(_fn(event))

        return (
            _env_reward,
            prov.get("reward_name", "?"),
            True,
            prov.get("spec_text") or prov.get("objective") or "",
            "env_reward",
        )
    # Frozen for the env's own reward but no recompilable code -> grade under the
    # env default reward (reward_function=None) with objective=env_reward, matching
    # how the base combiner was composed. Detect via the frozen ignore_pref flag
    # and/or a reward_provenance that names an env reward.
    frozen_env_reward = bool(meta.get("ignore_pref")) or bool(prov.get("reward_name"))
    if frozen_env_reward:
        return (
            None,
            prov.get("reward_name", "DefaultRewardFunction"),
            True,
            prov.get("spec_text") or prov.get("objective") or "",
            "env_reward",
        )
    return None, None, False, None, "scalarize"


def _select_skills(
    combiner: LLMCombiner,
    skills: Dict[str, Skill],
    scenario: Scenario,
    reward_function=None,
    *,
    threshold: float = 0.01,
    capture: int = 400,
    log: Callable[[str], None] = print,
) -> List[str]:
    """One capture rollout -> the skills the combiner ACTUALLY selects on ``scenario``.

    Runs the frozen combiner once on the concrete scenario with obs-capture on, then
    reads :meth:`LLMCombiner.fleet_pick_fractions` at the scenario's own preference.
    Returns the skill names whose fleet-selection fraction is ``>= threshold``, in
    descending fraction order. These are the fine-tune scope (§ approved: combiner +
    only the skills it uses here).
    """
    combiner.reset_telemetry()
    combiner.enable_capture(capture)
    env = build_env(scenario, reward_function=reward_function)
    ctrl = DispatchController(combiner, skills=skills)
    rollout(env, ctrl, scenario.preference, seed=scenario.seed)
    fracs = combiner.fleet_pick_fractions(scenario.preference)
    picked = sorted(
        (n for n, f in fracs.items() if f >= threshold),
        key=lambda n: fracs[n],
        reverse=True,
    )
    log(f"[select] fleet pick fractions: "
        f"{ {n: round(f, 3) for n, f in sorted(fracs.items(), key=lambda kv: -kv[1])} }")
    log(f"[select] fine-tuning {len(picked)} skill(s) (>= {threshold:g}): {picked}")
    return picked


def _load_skill_meta(name: str, skills_dir: str) -> Optional[Dict]:
    """Read a frozen skill's ``.meta.json`` (its code/fitness_code/objective) if present.

    Handwritten seeds (revenue/service/enroute) have no frozen meta on disk, so this
    returns ``None`` for them -- they cannot be warm-started (no self-authored
    fitness yardstick to hill-climb under) and are skipped by the caller.
    """
    meta_path = os.path.join(skills_dir, f"{name}.meta.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# The orchestrator.                                                            #
# --------------------------------------------------------------------------- #
def finetune_to_spec(
    spec: ScenarioSpec,
    *,
    base_combiner_name: str = "state_aware_pref_slider",
    model: Optional[str] = None,
    client=None,
    evolved_skills_dir: str = EvolvedSkillsDir,
    ranges: Optional[ScenarioRanges] = None,
    generations_combiner: int = 2,
    generations_skill: int = 1,
    lam: int = 1,
    skill_pick_threshold: float = 0.01,
    temperature: float = 0.9,
    out_root: str = FinetunedRoot,
    log: Callable[[str], None] = print,
) -> FinetuneResult:
    """Warm-start fine-tune the generalist policy to a (possibly PARTIAL) ``spec``.

    The spec pins whatever axes the operator gave (fleet / capacity / speed /
    regime / preference / reward) and DOMAIN-RANDOMIZES the rest: any axis left
    ``None`` is drawn per-scenario from ``ranges`` (default :class:`ScenarioRanges`),
    and the fine-tune evolves over that whole ``spec.n_scenarios`` batch. With every
    axis pinned and ``n_scenarios==1`` this reduces EXACTLY to the original
    single-concrete-scenario fine-tune.

    Loads the generalist basis + ``base_combiner_name``, discovers which skills the
    combiner uses on the batch, fine-tunes those skills then the combiner via
    warm-start (seeded from the frozen code, short improve loop over the batch), and
    freezes the specialised artifacts to ``<out_root>/<tag>/{skills,combiners}/`` --
    leaving the generalist basis on disk untouched. Returns a :class:`FinetuneResult`.

    ``client`` may be injected (tests / fakes); otherwise a real client is built from
    :class:`LLMConfig` (key read from ``YIBU_API_KEY`` / git-ignored ``.env`` only).
    """
    # --- Client (key fail-fast; never written/printed). ------------------- #
    if client is None:
        cfg = LLMConfig()
        if model:
            cfg.model = model
        client = make_llm_client(cfg)

    # --- Generalist basis + base combiner. -------------------------------- #
    skills, cards = load_basis(include_evolved=True, evolved_dir=evolved_skills_dir)
    combiner, base_meta = load_frozen_combiner(
        base_combiner_name, skill_names=tuple(skills)
    )
    log(f"[base] combiner={base_combiner_name!r} over basis {list(skills)}")

    # --- Objective: mirror the base combiner's reward contract. ----------- #
    reward_function, reward_name, ignore_pref, reward_spec, objective = (
        _reward_from_combiner_meta(base_meta)
    )
    if objective == "env_reward":
        log(f"[base] objective=env_reward under reward {reward_name!r} "
            f"(ignore_pref={ignore_pref}"
            + ("" if reward_function is not None else ", env default reward_function")
            + ")")
    else:
        log(f"[base] objective=scalarize under scenario preference")

    # --- Materialise the fine-tune batch: pinned axes fixed, rest random. -- #
    rg = ranges or ScenarioRanges(order_limit=spec.order_limit)
    scenarios = spec.build_scenarios(ranges=rg)
    rep = scenarios[0]  # representative scenario (profile + skill-select rollout)
    tag = spec_tag(spec, rep)
    skills_out = os.path.join(out_root, tag, "skills")
    combiners_out = os.path.join(out_root, tag, "combiners")
    if spec.randomized:
        log(f"[spec] pinned={spec.pinned} | RANDOMIZED={spec.randomized} "
            f"over {len(scenarios)} scenario(s)")
        for i, sc in enumerate(scenarios):
            log(f"[spec]   scenario[{i}] = {sc.label()}")
    else:
        log(f"[spec] fully concrete scenario = {rep.label()}")

    # --- Env profile for the prompts (built on the representative scenario). #
    # The profile advertises the FULL randomization envelope (ranges) so the LLM
    # sees it is specialising to a family, not overfitting one exact point.
    env = make_nyc_env(
        seed=rep.seed, regime=rep.regime, split=rep.split,
        num_drivers=rep.num_drivers, order_limit=rep.order_limit,
    )
    env.reset(seed=rep.seed)

    def dist(a, b):
        return env.network.shortest_path(a, b).travel_time

    phi = GlobalStats.from_observations(env._build_observations(), dist=dist)
    env_profile = encode_env_profile(
        env, phi, rep.regime, rep.split, random.Random(rep.seed),
        dist=dist, scenario=rep, ranges=rg, prev_windows=(1, 2),
    )

    # --- Scope: which skills does the combiner actually use here? ---------- #
    # Selected on the representative scenario (one capture rollout); the fine-tune
    # then evolves those over the whole batch.
    selected = _select_skills(
        combiner, skills, rep, reward_function=reward_function,
        threshold=skill_pick_threshold, log=log,
    )

    # --- Baseline: the frozen generalist combiner's fitness on THE BATCH. --- #
    # Measured on the untouched generalist skills (before any fine-tune), so the
    # before/after fitness delta is apples-to-apples with the combiner objective.
    base_frames = scenario_norm_frames(
        skills, scenarios, reward_function=reward_function
    )
    base_eval = evaluate_combiner_scenarios(
        combiner, skills, scenarios, base_frames,
        objective=objective, reward_function=reward_function,
    )
    log(f"[base] generalist combiner fitness on batch = {base_eval.fitness:.4g}")

    result = FinetuneResult(
        tag=tag, scenario=rep, base_combiner_name=base_combiner_name,
        combiner_name=base_combiner_name, skills_dir=skills_out,
        combiners_dir=combiners_out, selected_skills=list(selected),
        reward_name=reward_name, ignore_pref=ignore_pref, objective=objective,
        base_fitness=base_eval.fitness,
        pinned=dict(spec.pinned), randomized_axes=list(spec.randomized),
        n_scenarios=len(scenarios), scenarios=list(scenarios),
    )

    # --- Fine-tune the selected skills FIRST (combiner climbs on top). ----- #
    for name in selected:
        skmeta = _load_skill_meta(name, evolved_skills_dir)
        if skmeta is None:
            log(f"[skill {name}] no frozen meta (handwritten seed) -> skip warm-start")
            continue
        if not skmeta.get("fitness_code"):
            log(f"[skill {name}] frozen meta has no fitness_code -> skip warm-start")
            continue
        reference = skills[name]  # scale-free Δ baseline = the frozen generalist skill
        log(f"[skill {name}] warm-start fine-tune "
            f"(gens={generations_skill}, lam={lam}) over {len(scenarios)} scenario(s) ...")
        cand = evolve_one_skill(
            client, env_profile,
            scenarios=scenarios,
            reference=reference,
            seed_code=skmeta["code"],
            seed_fitness_code=skmeta["fitness_code"],
            seed_objective=skmeta.get("objective"),
            seed_meta=skmeta,
            generations=generations_skill, lam=lam,
            seed=rep.seed, temperature=temperature, log=log,
        )
        path = freeze_skill(cand, out_dir=skills_out, regime="finetune")
        # Swap the improved skill into the live basis so the combiner fine-tune and
        # its frames are measured on the specialised skill.
        from pref_dispatch.llm.basis import _load_evolved_module
        skills[name] = _load_evolved_module(path, name)
        result.finetuned_skills.append(name)
        result.skill_paths[name] = path
        log(f"[skill {name}] frozen -> {path}")

    # --- Fine-tune the combiner ON the (improved) skills. ----------------- #
    frames = scenario_norm_frames(skills, scenarios, reward_function=reward_function)
    log(f"[combiner] warm-start fine-tune "
        f"(gens={generations_combiner}, lam={lam}) over {len(scenarios)} scenario(s) ...")
    best = evolve_combiner(
        client, env_profile, skills, cards, [], None,
        scenarios=scenarios, scenario_frames=frames,
        reward_spec=reward_spec, reward_function=reward_function,
        ignore_pref=ignore_pref, objective=objective,
        seed_code=base_meta["code"], seed_meta=base_meta,
        generations=generations_combiner, lam=lam,
        seed=rep.seed, temperature=temperature, log=log,
    )
    result.finetuned_fitness = best.evaluation.fitness
    result.combiner_name = best.name

    reward_provenance = base_meta.get("reward_provenance")
    reward_snapshot = base_meta.get("reward_snapshot")
    combiner_path = freeze_combiner(
        best, out_dir=combiners_out,
        reward_snapshot=reward_snapshot, reward_provenance=reward_provenance,
        ignore_pref=ignore_pref,
    )
    result.combiner_path = combiner_path
    log(f"[combiner] frozen -> {combiner_path}")
    log(f"[done] base fitness {result.base_fitness:.4g} -> "
        f"fine-tuned {result.finetuned_fitness:.4g} on {tag}")
    return result


def finetune_to_scenario(
    scenario: Scenario,
    *,
    base_combiner_name: str = "state_aware_pref_slider",
    model: Optional[str] = None,
    client=None,
    evolved_skills_dir: str = EvolvedSkillsDir,
    generations_combiner: int = 2,
    generations_skill: int = 1,
    lam: int = 1,
    skill_pick_threshold: float = 0.01,
    temperature: float = 0.9,
    out_root: str = FinetunedRoot,
    log: Callable[[str], None] = print,
) -> FinetuneResult:
    """Backward-compatible wrapper: fine-tune to ONE fully-concrete ``scenario``.

    Equivalent to :func:`finetune_to_spec` with every axis pinned from ``scenario``
    and ``n_scenarios=1`` -- byte-identical to the original Phase-3 behaviour.
    """
    spec = ScenarioSpec(
        num_drivers=scenario.num_drivers,
        driver_capacity=scenario.driver_capacity,
        speed_kmh=scenario.speed_kmh,
        regime=scenario.regime,
        pref_revenue=scenario.pref_revenue,
        order_limit=scenario.order_limit,
        split=scenario.split,
        seed=scenario.seed,
        n_scenarios=1,
    )
    return finetune_to_spec(
        spec, base_combiner_name=base_combiner_name, model=model, client=client,
        evolved_skills_dir=evolved_skills_dir,
        generations_combiner=generations_combiner,
        generations_skill=generations_skill, lam=lam,
        skill_pick_threshold=skill_pick_threshold, temperature=temperature,
        out_root=out_root, log=log,
    )
