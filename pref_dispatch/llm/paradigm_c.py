"""Paradigm C (§5.5): the ONLINE LLM upper bound + its compute cost.

Paradigm B (the main method) evolves a frozen Python combiner OFFLINE, so online
cost is ~0 LLM calls. Paradigm C is the contrast: it keeps the SAME frozen skills
and the SAME matcher, but at **every decision step** it asks the LLM which skill
each driver should use, given the live pruned state. C is expensive (O(steps) LLM
calls) and is run only for a few episodes / sampled steps -- its role is to report
the *effect upper bound* and the *per-step token/latency cost* against B's near-
zero online cost (proposal 5.2's compute-vs-effect table).

Design (aligns with §5.5):

* ``encode_step_state`` compresses the current step to a bounded-token prompt: a
  small set of driver rows (zone, state, onboard slack) and, per driver, only the
  ``top_k`` spatially-nearest candidate orders (reusing the exact
  :func:`pref_dispatch.matching._knn_candidates` prune the matcher uses), each as
  (zone, trip-minutes, pickup-minutes). Drivers beyond ``max_drivers`` are sampled
  so the token budget is bounded on a 1000-car fleet.
* ``OnlineLLMController`` calls the LLM once per step, parses a
  ``{driver_id: skill_name}`` assignment, and turns it into the per-driver one-hot
  weights the existing matcher consumes -- identical execution path to B, only the
  weight SOURCE differs (live LLM vs frozen function).
* It defends against every failure mode (bad JSON, unknown skill, missing driver)
  by falling back to a default skill, and it METERS real tokens (from the client's
  ``last_usage``) and wall-clock latency per step so the compute table is honest.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from pref_dispatch.combiner import Combiner
from pref_dispatch.global_stats import GlobalStats
from pref_dispatch.llm.encode import _load_zones, _nearest_zone
from pref_dispatch.llm.extract import extract_json
from pref_dispatch.matching import _knn_candidates
from pref_dispatch.preference import Preference

Coord = Tuple[float, float]
Dist = Callable[[Coord, Coord], float]


# --------------------------------------------------------------------------- #
# Step-state encoding (bounded tokens).                                        #
# --------------------------------------------------------------------------- #
def _driver_state(driver_obs: Dict) -> str:
    """One-word live driver state (idle / loaded / pressed), for the prompt."""
    details = driver_obs["self"]["assigned_order_details"]
    if not details:
        return "idle"
    etas = [d["eta"] for d in details if d.get("eta") is not None]
    if etas and min(etas) <= 5.0:
        return "pressed"
    return "loaded"


def encode_step_state(
    observations: Dict[int, Dict],
    phi: GlobalStats,
    pref: Preference,
    skill_cards: Sequence[Dict],
    dist: Dist,
    *,
    top_k: int = 5,
    max_drivers: int = 20,
    zones: Optional[Sequence[Tuple[str, Coord]]] = None,
) -> Tuple[str, List[int]]:
    """Compress the live step into a bounded-token prompt.

    Returns ``(user_prompt, driver_ids)`` where ``driver_ids`` is the (possibly
    sampled) set of drivers the LLM is asked to assign -- the caller defaults any
    omitted driver. Only each driver's ``top_k`` nearest pending orders are shown
    (same prune as the matcher), and at most ``max_drivers`` drivers, so the token
    cost is bounded regardless of fleet size.
    """
    if zones is None:
        zones = _load_zones()

    any_obs = next(iter(observations.values()))
    pending = list(any_obs["pending_orders"])

    # Deterministic driver subset: the ``max_drivers`` with the most nearby
    # demand get shown (they are where the decisions matter). No RNG -> the
    # sampled compute table is reproducible.
    near = _knn_candidates(observations, pending, top_k)
    driver_ids = list(observations.keys())
    if len(driver_ids) > max_drivers:
        driver_ids.sort(key=lambda d: -len(near.get(d, ())))
        driver_ids = driver_ids[:max_drivers]

    scale = max(phi.mean_solo_time, 1e-6)
    lines: List[str] = []
    for did in driver_ids:
        obs = observations[did]
        loc = obs["self"]["location"]
        zone = _nearest_zone(loc, zones) if zones else "?"
        state = _driver_state(obs)
        cand_txt: List[str] = []
        for oi in near.get(did, [])[:top_k]:
            o = pending[oi]
            trip = dist(o["origin"], o["destination"]) / scale
            pick = dist(loc, o["origin"]) / scale
            oz = _nearest_zone(o["origin"], zones) if zones else "?"
            cand_txt.append(
                f"{{id:{o['order_id']}, from:{oz}, trip:{trip:.2f}, pickup:{pick:.2f}}}"
            )
        cands = "; ".join(cand_txt) if cand_txt else "(no nearby order)"
        lines.append(f"  driver {did} [{state} @ {zone}]: {cands}")

    skills = ", ".join(m.get("skill_name", m.get("name", "?")) for m in skill_cards)
    cards = "\n".join(
        f"  - {m.get('skill_name', m.get('name','?'))}: {m.get('objective','')}"
        for m in skill_cards
    )
    prompt = (
        "# LIVE DISPATCH STEP\n"
        f"time={phi.time:.0f}min, pending={phi.num_pending}, idle_drivers="
        f"{phi.num_idle}, demand_pressure={phi.demand_pressure:.2f}. "
        f"Trip/pickup are in units of mean_solo_time (~{phi.mean_solo_time:.1f} min).\n\n"
        f"PLATFORM PREFERENCE: revenue={pref['revenue']:.2f}, service={pref['service']:.2f}.\n\n"
        "FROZEN SKILLS you may assign (choose ONE per driver):\n"
        f"{cards}\n\n"
        "DRIVERS (state @ zone : nearby candidate orders):\n"
        + "\n".join(lines)
        + "\n\nAssign each driver the single best frozen skill for THIS step under "
        "the platform preference (revenue-heavy -> favour the revenue skill for "
        "idle cars near long fares; service-heavy -> favour service; protect "
        "pressed cars with enroute). Respond with ONE JSON object:\n"
        '{"assignments": {"<driver_id>": "<skill_name>", ...}, '
        '"reasoning": "<one sentence explaining the overall choice this step>"}\n'
        f"Use only these skill names: {skills}."
    )
    return prompt, driver_ids


_ONLINE_SYSTEM = (
    "You are an online ride-POOLING dispatcher. Each step you assign every listed "
    "driver ONE frozen scoring skill, adapting to the platform preference and each "
    "driver's live state. You always answer with exactly one JSON object of "
    "assignments plus a one-sentence reasoning (interpretability is required)."
)


# --------------------------------------------------------------------------- #
# The online controller (paradigm C).                                          #
# --------------------------------------------------------------------------- #
@dataclass
class StepMeter:
    """Per-step compute telemetry for the B/C table."""

    prompt_tokens: List[int] = field(default_factory=list)
    completion_tokens: List[int] = field(default_factory=list)
    latencies_s: List[float] = field(default_factory=list)
    n_calls: int = 0
    n_fallback_steps: int = 0  # steps whose LLM reply was unusable (full default)

    def record(self, usage: Optional[dict], latency_s: float) -> None:
        self.n_calls += 1
        self.latencies_s.append(latency_s)
        if usage:
            if usage.get("prompt_tokens") is not None:
                self.prompt_tokens.append(int(usage["prompt_tokens"]))
            if usage.get("completion_tokens") is not None:
                self.completion_tokens.append(int(usage["completion_tokens"]))

    def summary(self) -> Dict[str, float]:
        def _mean(xs):
            return sum(xs) / len(xs) if xs else 0.0

        return {
            "llm_calls": self.n_calls,
            "mean_prompt_tokens": _mean(self.prompt_tokens),
            "mean_completion_tokens": _mean(self.completion_tokens),
            "mean_total_tokens": _mean(self.prompt_tokens) + _mean(self.completion_tokens),
            "mean_latency_s": _mean(self.latencies_s),
            "total_latency_s": sum(self.latencies_s),
            "fallback_steps": self.n_fallback_steps,
        }


class OnlineLLMController(Combiner):
    """Paradigm-C combiner: query the LLM once per step for skill assignments.

    Unlike a paradigm-B :class:`Combiner`, this holds no frozen function -- it
    calls the live client. It caches the current step's assignment so the
    matcher's per-driver ``weights_for`` calls within a step reuse ONE LLM reply
    (the matcher iterates drivers; we must not call the LLM per driver).

    ``begin_step`` MUST be called once per env step before the matcher runs; the
    :class:`OnlineDispatchController` below wires that automatically.
    """

    def __init__(
        self,
        client,
        skill_cards: Sequence[Dict],
        *,
        default_skill: Optional[str] = None,
        top_k: int = 5,
        max_drivers: int = 20,
        temperature: float = 0.0,
        meter: Optional[StepMeter] = None,
        log_reasoning: Optional[Callable[[str], None]] = None,
    ):
        self.client = client
        self.skill_cards = list(skill_cards)
        self.skill_names = tuple(
            m.get("skill_name", m.get("name", "?")) for m in self.skill_cards
        )
        self.default_skill = default_skill or self.skill_names[0]
        self.top_k = top_k
        self.max_drivers = max_drivers
        self.temperature = temperature
        self.meter = meter or StepMeter()
        self.log_reasoning = log_reasoning
        self._zones = _load_zones()
        self._assignment: Dict[int, str] = {}

    def begin_step(
        self,
        observations: Dict[int, Dict],
        phi: GlobalStats,
        pref: Preference,
        dist: Dist,
    ) -> None:
        """Query the LLM for this step's per-driver skill assignment."""
        prompt, _driver_ids = encode_step_state(
            observations, phi, pref, self.skill_cards, dist,
            top_k=self.top_k, max_drivers=self.max_drivers, zones=self._zones,
        )
        t0 = time.perf_counter()
        try:
            raw = self.client.complete(
                _ONLINE_SYSTEM, prompt, temperature=self.temperature
            )
        except Exception:  # noqa: BLE001 -- a failed call must not kill the rollout
            self.meter.record(getattr(self.client, "last_usage", None),
                              time.perf_counter() - t0)
            self.meter.n_fallback_steps += 1
            self._assignment = {}
            return
        latency = time.perf_counter() - t0
        self.meter.record(getattr(self.client, "last_usage", None), latency)

        assignment = self._parse(raw)
        if not assignment:
            self.meter.n_fallback_steps += 1
        elif self.log_reasoning is not None:
            self.log_reasoning(assignment.get("__reasoning__", ""))
        self._assignment = {
            k: v for k, v in assignment.items() if k != "__reasoning__"
        }

    def _parse(self, raw: str) -> Dict:
        """Parse the LLM reply into ``{driver_id: skill_name}`` (+reasoning).

        Unknown skills / non-int ids are dropped; an empty result signals the
        caller to fall back for the whole step.
        """
        try:
            obj = extract_json(raw)
        except Exception:  # noqa: BLE001
            return {}
        assignments = obj.get("assignments", obj) if isinstance(obj, dict) else {}
        out: Dict = {}
        if isinstance(assignments, dict):
            for k, v in assignments.items():
                if v in self.skill_names:
                    try:
                        out[int(k)] = v
                    except (TypeError, ValueError):
                        continue
        if isinstance(obj, dict) and obj.get("reasoning"):
            out["__reasoning__"] = str(obj["reasoning"])
        return out

    # -- Combiner contract (consumed by the existing matcher) ------------- #
    def weights_for(self, driver_obs, phi_ep, phi_step, w=None):
        did = driver_obs["self"]["driver_id"]
        skill = self._assignment.get(did, self.default_skill)
        return {skill: 1.0}

    def classify(self, driver_obs, phi_ep, phi_step):
        did = driver_obs["self"]["driver_id"]
        return self._assignment.get(did, self.default_skill)
