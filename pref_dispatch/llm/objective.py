"""A5 -- translate a natural-language objective into ``w``: a callable reward fn.

The final-version episode entry (:func:`pref_dispatch.evaluate.rollout`) carries the
episode objective ``w`` on the episode-static ``phi_ep`` and hands it to the combiner
and repositioner (never the skills). ``w`` is an LLM-authored **callable reward
function** ``w(event) -> float`` over the env's per-step ``event`` dict -- the inverse
of :func:`pref_dispatch.llm.reward_spec.describe_reward` (which renders a reward INTO
text; here we translate a preference INTO a reward).

This is done OFFLINE, ONCE per episode, by the caller before the rollout loop, so the
per-step dispatch stays a frozen-function forward pass with no online LLM call (the
paradigm-B contract). The result is then frozen for the whole episode.

The heavy lifting -- prompt, chain-of-thought gate, AST sandbox compile + validate,
repair loop -- already lives in :func:`pref_dispatch.llm.evolve_reward.author_reward`,
which returns an :class:`~pref_dispatch.llm.evolve_reward.AuthoredReward` whose ``fn``
is the env-shape ``(driver_id, event) -> float`` reward. This module is the thin A5
adapter that:

* accepts the same preference FORMS (NL brief / weight dict / concrete
  ``RewardFunction``);
* returns the **single-argument** ``w(event) -> float`` the combiner/repositioner and
  ``phi_ep.reward_fn`` consume (driver identity is not part of the objective surface),
  alongside the provenance meta + NL explanation the artefact records.

Security: never reads or writes an API key. The ``client`` holds its own env-only key
(:mod:`pref_dispatch.llm.client`); this module only calls ``complete`` through
``author_reward``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

from pref_dispatch.llm.client import LLMClient
from pref_dispatch.llm.evolve_reward import (
    AuthoredReward,
    RewardAuthoringError,
    author_reward,
)
from pref_dispatch.llm.reward_spec import RewardLike

RewardFn = Callable[[Dict], float]

__all__ = ["ObjectiveW", "translate_objective", "RewardAuthoringError"]


@dataclass
class ObjectiveW:
    """A translated episode objective, ready to carry on ``phi_ep``.

    ``w`` is the single-argument ``w(event) -> float`` callable the combiner +
    repositioner read (and MAY call to self-derive). ``label`` is a short human
    brief for logging / ``phi_ep.objective_label``. ``meta`` carries the full
    provenance the artefact records (CoT reward-understanding, code, spec_text),
    lifted straight from the underlying :class:`AuthoredReward`.
    """

    w: RewardFn
    label: str
    meta: Dict

    @property
    def explanation(self) -> str:
        """The NL explanation of what this objective rewards (artefact provenance)."""
        return self.meta.get("reward_understanding", "") or self.meta.get(
            "description", ""
        )

    @property
    def code(self) -> Optional[str]:
        return self.meta.get("code")

    @property
    def is_authored(self) -> bool:
        return bool(self.meta.get("authored"))


def _as_event_fn(authored: AuthoredReward) -> RewardFn:
    """Recover the single-arg ``w(event)`` from an env-shape authored reward.

    ``AuthoredReward.fn`` is ``(driver_id, event) -> float`` (what the env grades
    with). The objective surface handed to the combiner/repositioner does NOT
    include driver identity -- ``w`` is a pure function of the per-step ``event`` --
    so we bind a fixed driver id. The authored reward ignores the id by contract
    (see :func:`pref_dispatch.llm.evolve_reward._adapt_event_reward`), so this is a
    faithful, side-effect-free adapter.
    """
    fn = authored.fn

    def w(event: Dict) -> float:
        return float(fn(0, event))

    return w


def translate_objective(
    client: LLMClient,
    objective: RewardLike,
    *,
    n_repair: int = 2,
    temperature: float = 0.7,
    log: Callable[[str], None] = print,
) -> ObjectiveW:
    """Translate a preference/objective into ``w`` (a callable reward fn).

    ``objective`` may be a natural-language brief, a metric-weight dict, or a
    concrete :class:`~ride_gym.rewards.RewardFunction` (in which case authoring is
    skipped and the instance IS the objective). Returns an :class:`ObjectiveW`
    whose ``w`` is the ``w(event) -> float`` callable to pass as
    ``rollout(reward_fn=objective_w.w, objective_label=objective_w.label)`` and
    which is then frozen on ``phi_ep`` for the whole episode.

    Raises :class:`RewardAuthoringError` if no valid reward can be authored after
    the repair budget (an unreliable objective must fail loudly, not silently run
    objective-blind).
    """
    authored = author_reward(
        client, objective, n_repair=n_repair, temperature=temperature, log=log
    )
    label = str(authored.meta.get("objective") or authored.name or "objective").strip()
    return ObjectiveW(w=_as_event_fn(authored), label=label, meta=authored.meta)
