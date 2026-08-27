"""Pure-Phase-1 evolution (proposal step 4): evolve ONE scorer under the FIXED
default MARL reward -- no self-authored fitness, no upper combiner.

This is the ablation rung that answers "what does the LLM's lower layer buy on
its own, judged by the SAME objective the MARL baselines optimise?". It mirrors
:func:`pref_dispatch.llm.evolve.evolve_one_skill` but with two deliberate
differences that define the arm:

1. **The reward is FIXED, not self-authored.** Every generation is scored by the
   researcher-fixed fitness ``lambda m: m["income_mean"]`` -- the realised
   fleet-mean of ``ride_gym.rewards.DefaultRewardFunction``, i.e. the exact
   reward IDDQN / MF-DDQN / BMG-Q are trained on. There is no ``fitness_code``
   field in the contract.
2. **A chain-of-thought interpretability gate.** The model must FIRST explain,
   in ``reward_understanding``, what the given reward incentivises, before it is
   allowed to write the policy -- the "explain the reward first" step the user
   asked for.

The winner is frozen to ``pref_dispatch/evolved/pure_phase1/`` (a directory kept
separate from the QD basis so this ablation never pollutes ``load_basis``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

from pref_dispatch.llm.client import LLMClient
from pref_dispatch.llm.extract import extract_json, require_explanation
from pref_dispatch.llm.fitness_eval import (
    DEFAULT_REGIMES,
    EVAL_NUM_DRIVERS,
    EVAL_ORDER_LIMIT,
    SkillEval,
    evaluate_skill,
)
from pref_dispatch.llm.prompts.pure_phase1 import (
    REQUIRED_EXPLANATION_FIELDS,
    REQUIRED_FIELDS,
    build_pure_phase1_prompt,
)
from pref_dispatch.llm.sandbox import (
    CompiledSkill,
    SandboxError,
    compile_skill,
    validate_skill,
)

FrozenDir = os.path.join("pref_dispatch", "evolved", "pure_phase1")

# The researcher-FIXED reward: realised fleet mean of DefaultRewardFunction, the
# exact objective the MARL baselines are trained on. This is the whole point of
# the arm, so it lives here as a named constant, not a magic lambda at a call site.
def default_marl_reward(metrics: Dict[str, float]) -> float:
    """Fleet-mean per-driver DefaultRewardFunction reward (the MARL objective)."""
    return float(metrics["income_mean"])


class EvolutionError(RuntimeError):
    """Raised when a generation cannot produce any valid candidate."""


@dataclass
class PureCandidate:
    """A validated pure-Phase-1 scorer plus provenance and measured fitness."""

    meta: Dict  # skill_name, objective, description, reward_understanding, code, gen
    skill: CompiledSkill
    evaluation: Optional[SkillEval] = None

    @property
    def name(self) -> str:
        return self.meta["skill_name"]

    @property
    def score_value(self) -> float:
        return self.evaluation.fitness_mean if self.evaluation is not None else float("-inf")


def _check_fields(obj: Dict) -> None:
    missing = [f for f in REQUIRED_FIELDS if not str(obj.get(f, "")).strip()]
    if missing:
        raise SandboxError(f"response missing required field(s): {missing}")
    require_explanation(obj, REQUIRED_EXPLANATION_FIELDS)


def _build_candidate(obj: Dict, gen: int) -> PureCandidate:
    _check_fields(obj)
    name = str(obj["skill_name"]).strip()
    skill = compile_skill(obj["code"], name=name)
    ok, why = validate_skill(skill)
    if not ok:
        raise SandboxError(f"skill invalid: {why}")
    meta = {
        "skill_name": name,
        "objective": obj["objective"].strip(),
        "description": obj["description"].strip(),
        "reward_understanding": obj["reward_understanding"].strip(),
        "code": obj["code"],
        "gen": gen,
    }
    return PureCandidate(meta=meta, skill=skill)


def _ask(client: LLMClient, prompt: Dict[str, str], temperature=None) -> Dict:
    raw = client.complete(prompt["system"], prompt["user"], temperature=temperature)
    return extract_json(raw)


def _propose_with_repair(
    client: LLMClient,
    build_prompt: Callable[[Optional[str]], Dict[str, str]],
    gen: int,
    *,
    n_repair: int = 2,
    temperature: Optional[float] = None,
) -> PureCandidate:
    feedback: Optional[str] = None
    last_err = ""
    for _ in range(n_repair + 1):
        prompt = build_prompt(feedback)
        try:
            obj = _ask(client, prompt, temperature=temperature)
            return _build_candidate(obj, gen)
        except Exception as e:  # noqa: BLE001 -- retry on any
            last_err = f"{type(e).__name__}: {e}"
            feedback = last_err
    raise EvolutionError(f"no valid candidate after repairs; last error: {last_err}")


def evolve_pure_phase1(
    client: LLMClient,
    env_profile: str,
    *,
    generations: int = 4,
    lam: int = 2,
    regimes: Sequence[str] = DEFAULT_REGIMES,
    split: str = "train",
    num_drivers: int = EVAL_NUM_DRIVERS,
    order_limit: Optional[int] = EVAL_ORDER_LIMIT,
    seed: int = 0,
    temperature: float = 0.9,
    log: Callable[[str], None] = print,
) -> PureCandidate:
    """Evolve ONE fleet-wide scorer under the FIXED default MARL reward.

    Same (1+lambda) hill-climb shape as :func:`evolve_one_skill`, but the fitness
    is fixed to :func:`default_marl_reward` at EVERY generation (including gen 0)
    -- the model never authors a yardstick, it only maximises the given one.
    """
    fitness_fn = default_marl_reward

    def _eval(cand: PureCandidate) -> SkillEval:
        return evaluate_skill(
            cand.skill, fitness_fn, regimes=regimes, split=split,
            num_drivers=num_drivers, order_limit=order_limit, seed=seed,
        )

    # --- Generation 0: propose (explain reward, then policy). -------------- #
    best = _propose_with_repair(
        client,
        lambda fb: build_pure_phase1_prompt(env_profile, repair_feedback=fb),
        gen=0, temperature=temperature,
    )
    best.evaluation = _eval(best)
    log(
        f"[gen 0] {best.name!r} objective={best.meta['objective']!r} "
        f"reward(income_mean)={best.evaluation.fitness_mean:.4g}"
    )
    log(f"        reward_understanding: {best.meta['reward_understanding']}")

    # --- Generations 1..G: improve the scorer under the fixed reward. ------ #
    for gen in range(1, generations + 1):
        for _ in range(lam):
            try:
                cand = _propose_with_repair(
                    client,
                    lambda fb: build_pure_phase1_prompt(
                        env_profile,
                        current_code=best.meta["code"],
                        current_fitness=best.evaluation.fitness_mean,
                        repair_feedback=fb,
                    ),
                    gen=gen, temperature=temperature,
                )
            except EvolutionError as e:
                log(f"[gen {gen}] candidate failed: {e}")
                continue
            cand.evaluation = _eval(cand)
            improved = cand.evaluation.fitness_mean > best.evaluation.fitness_mean
            log(
                f"[gen {gen}] {cand.name!r} reward={cand.evaluation.fitness_mean:.4g} "
                f"({'ACCEPT' if improved else 'reject'})"
            )
            if improved:
                best = cand

    return best


def freeze_pure_phase1(cand: PureCandidate, out_dir: str = FrozenDir,
                       *, regime: str = "multi") -> str:
    """Write the winning pure-Phase-1 scorer to disk as a runnable module + meta."""
    os.makedirs(out_dir, exist_ok=True)
    name = cand.name
    py_path = os.path.join(out_dir, f"{name}.py")
    meta_path = os.path.join(out_dir, f"{name}.meta.json")

    header = (
        f'"""Frozen pure-Phase-1 scorer: {name}\n\n'
        f"Objective: {cand.meta['objective']}\n\n"
        f"{cand.meta['description']}\n\n"
        f"Reward understanding (LLM CoT): {cand.meta['reward_understanding']}\n\n"
        f"Evolved under the FIXED default MARL reward (income_mean), gen "
        f"{cand.meta['gen']} (regime={regime}). No self-authored fitness, no upper\n"
        f'combiner: this ONE policy dispatches the whole fleet."""\n\n'
        "import math\n"
        "import numpy as np\n\n"
        "from pref_dispatch.skills import (\n"
        "    _feasible, _pickup_time, _solo_time, _onboard_slack,\n"
        ")\n\n\n"
    )
    with open(py_path, "w", encoding="utf-8") as f:
        f.write(header + cand.meta["code"].rstrip() + "\n")

    meta = dict(cand.meta)
    meta["regime"] = regime
    meta["fixed_reward"] = "income_mean (DefaultRewardFunction fleet mean)"
    if cand.evaluation is not None:
        meta["fitness_mean"] = cand.evaluation.fitness_mean
        meta["per_regime_fitness"] = cand.evaluation.per_regime_fitness
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return py_path
