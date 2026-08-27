"""Prompt builder for the PURE-Phase-1 ablation (proposal step 4).

This is the "one policy, fixed reward" rung of the effectiveness ladder. Unlike
ordinary Phase-1 skill discovery (:mod:`pref_dispatch.llm.prompts.skill_evolve`),
where the model *self-authors* a per-skill fitness, here the reward is FIXED by
the researcher: it is the exact ``DefaultRewardFunction`` that every MARL baseline
(IDDQN / MF-DDQN / BMG-Q) is trained on. The model does not invent a reward; it is
GIVEN the reward, must first explain it in natural language (a chain-of-thought
interpretability gate), and then evolve ONE fleet-wide scoring policy that
maximises that reward's realised fleet mean (``income_mean``).

The point of the arm: it isolates "what does the LLM's lower layer buy us on its
own, judged by the SAME objective the MARL methods optimise" -- no upper combiner,
no preference conditioning, no self-authored yardstick. It is the honest bridge
between the MARL policies and our two-layer method.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pref_dispatch.llm.prompts.common import (
    INTERPRETABILITY_RULE,
    SANDBOX_RULES,
    SIGNATURE_SPEC,
    few_shot_seeds,
)

# A faithful, human-readable statement of the FIXED default reward -- the exact
# per-driver, per-step reward every MARL baseline is trained on
# (``ride_gym.rewards.DefaultRewardFunction`` with the benchmark coefficients in
# ``benchmark/config.py``). The scorer is evolved to maximise this reward's
# realised fleet mean, so the model must understand what the reward pays for.
FIXED_REWARD_SPEC = """\
The reward is FIXED (you do NOT design it). It is the EXACT per-driver, per-step
reward that the reinforcement-learning baselines you are being compared against
are trained on. For a driver at one step it sums:

  + 1.00 * (number of orders newly ASSIGNED to this driver this step)
  + 0.01 * (for each newly assigned order) its solo (direct pickup->dropoff)
            trip time in minutes * its passenger count        [revenue / fare]
  - 0.04 * (for each newly assigned order) the EXTRA time beyond solo, i.e.
            predicted end-to-end service time minus solo time [service penalty]
  - 0.08 * signed re-routing impact on this driver's already-committed onboard
            orders (later deliveries cost, earlier ones credit) [detour penalty]
  (empty-move and idle penalties are zero in this configuration.)

Read what this reward WANTS from a dispatcher, in plain terms:
  - It pays first and foremost for ASSIGNING orders (the +1.0 bonus dominates):
    a driver that serves demand beats one that idles. Throughput matters most.
  - Among orders, it mildly prefers longer/served-minute-rich fares (the +0.01
    revenue term) but PENALISES orders whose end-to-end fulfilment drags far
    beyond their direct time (the -0.04 term) -- so it dislikes long pickups and
    heavy pooling delay.
  - It protects already-onboard riders: adding an order that detours committed
    deliveries is penalised (-0.08), so reckless pooling is discouraged.
The realised episode objective you are scored by is the FLEET MEAN of this
per-driver reward accumulated over the whole hour (``income_mean`` in the metrics
dict). Higher is better. Design the scorer to maximise it.
"""

# Output contract: NO fitness_code / fitness_rationale (the reward is fixed).
# Instead a mandatory ``reward_understanding`` CoT field: the model must show it
# understood the given reward BEFORE writing the policy.
OUTPUT_CONTRACT = """\
Respond with ONE JSON object and nothing else (no prose before/after, no markdown
outside the JSON). Schema:

{
  "reward_understanding": "<2-4 sentences, FIRST: in your own words, what the
                 FIXED reward above rewards and penalises, and what a
                 reward-maximising dispatcher must therefore do. This is a
                 chain-of-thought gate: reason about the objective before you
                 write the policy.>",
  "skill_name":  "<short snake_case id, e.g. default_reward_maximizer>",
  "objective":   "<ONE sentence: the dispatch policy you will use to maximise the
                 fixed reward>",
  "description": "<2-4 sentences: HOW the score logic maximises the fixed reward
                 -- what it prioritises assigning, what it declines, when it
                 waits (noop). Explain behaviour, not code.>",
  "code": "def score(driver_obs, order, phi_ep, phi_step):\\n    ...\\n\\ndef noop_score(driver_obs, phi_ep, phi_step):\\n    ..."
}

There is NO fitness field: the reward is fixed and given above; you only write the
scoring policy that maximises it.
"""

SYSTEM_PROMPT = """\
You are an expert in ride-POOLING fleet dispatch and reinforcement-learning
reward analysis. You are given a FIXED reward function (the same one competing RL
agents are trained on). You must first explain, in natural language, what that
reward incentivises, and then write ONE small, robust, interpretable Python
scoring policy that a whole fleet uses to MAXIMISE that reward. You always answer
with exactly one JSON object matching the requested schema.
"""


def build_pure_phase1_prompt(
    env_profile: str,
    *,
    current_code: Optional[str] = None,
    current_fitness: Optional[float] = None,
    repair_feedback: Optional[str] = None,
) -> Dict[str, str]:
    """Build the ``{"system","user"}`` prompt for the pure-Phase-1 arm.

    ``current_code`` / ``current_fitness`` are ``None`` on the first (propose)
    generation and set on improvement generations (rewrite the scorer to raise
    the FIXED reward's realised fleet mean).
    """
    parts: List[str] = []

    if current_code is None:
        parts.append(
            "# TASK\nWrite ONE fleet-wide dispatch scoring policy that MAXIMISES "
            "the fixed reward below. First explain the reward, then write the "
            "policy. There is no upper layer and no preference here: this single "
            "policy is used by every driver."
        )
    else:
        parts.append(
            "# TASK\nImprove your dispatch policy to score a HIGHER realised "
            "fleet-mean reward (income_mean) under the SAME fixed reward, keeping "
            "the same contract. Re-explain the (improved) behaviour."
        )
        if current_fitness is not None:
            parts.append(
                "# CURRENT BEST POLICY (measured fleet-mean reward = "
                f"{current_fitness:.4g})\n```python\n{current_code}\n```"
            )

    parts.append("# FIXED REWARD (given -- you do NOT change it)\n" + FIXED_REWARD_SPEC)
    parts.append("# ENVIRONMENT PROFILE\n" + env_profile)
    parts.append("# FUNCTION CONTRACT\n" + SIGNATURE_SPEC)
    parts.append("# SANDBOX RULES\n" + SANDBOX_RULES)
    parts.append("# SEED EXAMPLES (valid outputs and your starting point)\n"
                 + few_shot_seeds())

    if repair_feedback:
        parts.append(
            "# YOUR PREVIOUS ATTEMPT FAILED -- FIX THIS AND RESUBMIT\n"
            f"{repair_feedback}\n"
            "Return a corrected JSON object addressing exactly this error."
        )

    parts.append("# INTERPRETABILITY\n" + INTERPRETABILITY_RULE)
    parts.append("# OUTPUT FORMAT\n" + OUTPUT_CONTRACT)

    return {"system": SYSTEM_PROMPT, "user": "\n\n".join(parts)}


# Interpretability fields the extractor must find non-empty. ``reward_understanding``
# is the CoT gate specific to this arm (the model must show it read the reward).
REQUIRED_EXPLANATION_FIELDS = ("reward_understanding", "objective", "description")
# All required fields (NO fitness_code -- the reward is fixed and researcher-given).
REQUIRED_FIELDS = ("reward_understanding", "skill_name", "objective", "description", "code")
