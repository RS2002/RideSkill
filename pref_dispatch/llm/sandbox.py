"""Safely compile and validate LLM-authored code (skills, fitness, rewards).

Everything the LLM writes in paradigm B is Python that we ``exec`` and then run
thousands of times in the dispatch hot loop, so it must be sandboxed. Four kinds
of artefact pass through here (final-version two-layer signatures):

* **Skill** -- ``score(driver_obs, order, phi_ep, phi_step) -> float`` plus an
  optional ``noop_score(driver_obs, phi_ep, phi_step) -> float`` (§3.3). Executed
  per (driver, order) at match time. The travel-time closure is ``phi_ep.dist``
  (episode-static), so the skill reads it off ``phi_ep`` rather than a separate
  argument. Skills never see the objective ``w`` (they stay specialists).
* **Repositioner** -- ``reposition_scores(driver_obs, phi_ep, phi_step, kappa, w)
  -> {region_idx: float}`` (Feature 3, Phase 2) that authors the per-region base
  attractiveness for an idle driver's empty cruise. ``kappa`` is the shared
  per-region demand/supply state; ``w`` is the episode objective. Compiled
  independently of skills; executed per idle driver only when repositioning is on.
* **Fitness** -- ``fitness(metrics) -> float``, the LLM's *self-authored* reward
  over an episode-metrics dict (§4.6). The whole point is that it stays a cheap
  scalar function of the metrics dict (no rollout, no env, no LLM call) so
  evolution stays in paradigm B. The sandbox enforces "cheap" structurally: with
  imports and attribute access to internals forbidden, a fitness body can only do
  arithmetic over the dict it is handed.
* **Reward** -- ``reward(event) -> float``, the LLM's *self-authored* per-driver,
  per-step platform reward over the env's per-step ``event`` dict. This is how a
  natural-language / weights preference is TRANSLATED into a concrete objective:
  the authored reward is injected into the env as its ``reward_function`` and the
  cumulative fleet-mean value (``income_mean``) becomes the combiner's fitness.
  Same "cheap arithmetic over the handed-in dict" discipline as fitness.

Safety model (defence in depth, not a true security boundary -- we control the
model, but we still refuse obviously dangerous constructs so a hallucinated
``import os; os.system(...)`` cannot run):

1. **AST whitelist** -- reject ``import``, ``with``, ``global``, ``class``,
   generators, ``await``; reject dangerous builtins (``eval``/``exec``/``open``/
   ``__import__``/``getattr``/...); reject any ``_underscore`` attribute access
   (blocks ``__globals__``/``__class__`` escapes).
2. **Restricted globals** -- only a small safe-builtins set plus ``math`` and
   ``numpy as np`` and (for skills) the reusable primitives from
   :mod:`pref_dispatch.skills`.
3. **Runtime validation** -- actually call the compiled function on a synthetic
   input and require a finite float. On any failure we return a human-readable
   message the evolution loop feeds back to the LLM (or uses to fall back to the
   handwritten seed).
"""

from __future__ import annotations

import ast
import builtins
import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from pref_dispatch.budget import FairnessBudget
from pref_dispatch.global_stats import (
    EpisodeStats,
    GlobalStats,
    _nearest_region,
    od_matrix,
)
from pref_dispatch.reposition import RegionState
from pref_dispatch.skills import (
    _feasible,
    _onboard_slack,
    _pickup_time,
    _solo_time,
)

Coord = Tuple[float, float]
Dist = Callable[[Coord, Coord], float]


class SandboxError(ValueError):
    """Raised when code fails the AST whitelist or cannot be compiled."""


# --------------------------------------------------------------------------- #
# AST whitelist.                                                               #
# --------------------------------------------------------------------------- #
_FORBIDDEN_NODES: Tuple[type, ...] = (
    ast.Import,
    ast.ImportFrom,
    ast.With,
    ast.AsyncWith,
    ast.AsyncFunctionDef,
    ast.AsyncFor,
    ast.Await,
    ast.Global,
    ast.Nonlocal,
    ast.Yield,
    ast.YieldFrom,
    ast.ClassDef,
    ast.Delete,
)

# Builtins that could break out of the sandbox or touch the outside world.
_FORBIDDEN_NAMES = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "open",
        "__import__",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
        "input",
        "exit",
        "quit",
        "help",
        "breakpoint",
        "memoryview",
        "object",
        "super",
        "type",
        "classmethod",
        "staticmethod",
        "property",
        "__builtins__",
    }
)

# The only builtins skill/fitness bodies may reference.
_SAFE_BUILTIN_NAMES = (
    "abs min max sum len float int round sorted range enumerate zip map filter "
    "pow all any bool list dict tuple set frozenset reversed divmod isinstance "
    "print"
).split()
_SAFE_BUILTINS: Dict[str, object] = {
    n: getattr(builtins, n) for n in _SAFE_BUILTIN_NAMES
}


def _check_ast(tree: ast.AST) -> None:
    """Walk the tree and reject anything outside the whitelist."""
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_NODES):
            raise SandboxError(
                f"forbidden syntax: {type(node).__name__} is not allowed "
                "(no imports / classes / generators / global / with / await)."
            )
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            if node.id == "getattr":
                raise SandboxError(
                    "forbidden name: 'getattr' may not be used. "
                    "Read dict fields with d['key'] or d.get('key', default) "
                    "instead -- getattr raises AttributeError on dict keys anyway."
                )
            raise SandboxError(f"forbidden name: {node.id!r} may not be used.")
        # Block dunder / private attribute access (e.g. ``x.__globals__``), the
        # classic sandbox-escape vector.
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise SandboxError(
                f"forbidden attribute access: {node.attr!r} (leading underscore)."
            )


def _exec_restricted(code: str, extra: Dict[str, object]) -> Dict[str, object]:
    """Parse -> whitelist -> exec ``code`` in a restricted namespace.

    Returns the resulting namespace dict. Raises :class:`SandboxError` on a
    syntax error or a whitelist violation.
    """
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        raise SandboxError(f"syntax error: {e}") from e
    _check_ast(tree)
    g: Dict[str, object] = {
        "__builtins__": _SAFE_BUILTINS,
        "math": math,
        "np": np,
        **extra,
    }
    try:
        exec(compile(tree, "<sandbox>", "exec"), g)  # noqa: S102 -- whitelisted
    except Exception as e:  # noqa: BLE001 -- top-level body error
        raise SandboxError(f"error while defining functions: {e!r}") from e
    return g


# --------------------------------------------------------------------------- #
# Skill compilation + validation.                                             #
# --------------------------------------------------------------------------- #
# Primitives the LLM may call directly (declared in the prompt, §3.4).
_SKILL_PRIMITIVES: Dict[str, object] = {
    "_feasible": _feasible,
    "_pickup_time": _pickup_time,
    "_solo_time": _solo_time,
    "_onboard_slack": _onboard_slack,
}


@dataclass
class CompiledSkill:
    """A validated, executable skill (same call surface as the handwritten
    :class:`~pref_dispatch.skills.Skill`)."""

    name: str
    score: Callable[[Dict, Dict, EpisodeStats, GlobalStats], float]
    noop_score: Callable[[Dict, EpisodeStats, GlobalStats], float]
    # The source this was compiled from. An exec'd function cannot be pickled, so
    # a skill can only reach a worker process as source that the worker recompiles
    # through this same validator (see :mod:`pref_dispatch.llm.parallel`).
    code: Optional[str] = None

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"CompiledSkill({self.name})"


def compile_skill(code: str, name: str = "evolved") -> CompiledSkill:
    """Compile LLM skill ``code`` into a :class:`CompiledSkill`.

    ``code`` must define ``score(driver_obs, order, phi_ep, phi_step)``;
    ``noop_score(driver_obs, phi_ep, phi_step)`` is optional (a constant-0 no-op is
    supplied when absent, matching the base :class:`~pref_dispatch.skills.Skill`).
    Raises :class:`SandboxError`.
    """
    # Primitives are injected as globals so the body may call them directly.
    ns = _exec_restricted(code, dict(_SKILL_PRIMITIVES))
    score = ns.get("score")
    if not callable(score):
        raise SandboxError("skill code must define a callable `score`.")
    noop = ns.get("noop_score")
    if not callable(noop):
        def noop(driver_obs, phi_ep, phi_step):  # noqa: ARG001 -- default floor
            return 0.0
    return CompiledSkill(name=name, score=score, noop_score=noop, code=code)


# --------------------------------------------------------------------------- #
# Fitness compilation + validation.                                           #
# --------------------------------------------------------------------------- #
def compile_fitness(code: str) -> Callable[[Dict], float]:
    """Compile an LLM self-authored ``fitness(metrics) -> float``.

    No primitives, no env -- only ``math`` / ``np`` and safe builtins over the
    metrics dict, which is what keeps fitness a cheap paradigm-B scalar (§4.6).
    """
    ns = _exec_restricted(code, {})
    fn = ns.get("fitness")
    if not callable(fn):
        raise SandboxError("fitness code must define a callable `fitness`.")
    return fn


# --------------------------------------------------------------------------- #
# Runtime validation on synthetic inputs.                                     #
# --------------------------------------------------------------------------- #
def _synthetic_dist() -> Dist:
    """Cheap Euclidean-ish travel-time stand-in (minutes); strictly positive."""

    def dist(a: Coord, b: Coord) -> float:
        return 1.0 + 60.0 * math.dist(a, b)

    return dist


# A small region layout reused by the skill / repositioner probes so phi_ep has a
# non-trivial static scale and adjacency.
_PROBE_CENTRES: Tuple[Coord, ...] = (
    (-73.98, 40.75), (-73.95, 40.79), (-73.92, 40.81), (-73.90, 40.78),
)
_PROBE_NEIGHBOURS: Tuple[Tuple[int, ...], ...] = ((1,), (0, 2), (1, 3), (2,))

# A synthetic "previous window" for the probe OD matrix. Deliberately LOPSIDED --
# region 0 sends far more than it receives -- so a scorer that reads the OD matrix
# hits real non-zero, non-uniform numbers here instead of an all-equal matrix that
# would hide a division by a zero cell. Bucketed through the same
# :func:`~pref_dispatch.global_stats.od_matrix` the live path uses.
_PROBE_PREV_ORDERS: Tuple[Dict, ...] = tuple(
    {"origin": o, "destination": d, "num_passengers": 1}
    for o, d in (
        [(_PROBE_CENTRES[0], _PROBE_CENTRES[1])] * 5
        + [(_PROBE_CENTRES[0], _PROBE_CENTRES[3])] * 3
        + [(_PROBE_CENTRES[1], _PROBE_CENTRES[0])] * 2
        + [(_PROBE_CENTRES[2], _PROBE_CENTRES[2])] * 1
    )
)


def _probe_region(point: Coord) -> int:
    """Region of ``point`` in the probe layout, by the ONE partition rule.

    Computed, never hand-written, so the probe's ``origin_region`` /
    ``destination_region`` / ``current_region`` cannot drift away from what the
    env would actually ship if someone nudges a probe coordinate.
    """
    import numpy as np

    return int(_nearest_region(
        np.asarray([point], dtype=float), np.asarray(_PROBE_CENTRES, dtype=float)
    )[0])


def _synthetic_phi(
    *,
    reward_fn: Optional[Callable[[Dict], float]] = None,
    fairness_strength: float = 0.0,
) -> Tuple[EpisodeStats, GlobalStats]:
    """A schema-faithful ``(phi_ep, phi_step)`` pair for smoke runs.

    ``phi_ep`` carries the travel-time closure + static region layout + a non-empty
    previous-window OD matrix; ``phi_step`` carries the live aggregates plus
    non-empty per-region kappa arrays, so a scorer that reads the OD matrix or
    kappa exercises a real path here rather than only in a live rollout.
    """
    dist = _synthetic_dist()
    od_count, od_out, od_in, od_orders = od_matrix(_PROBE_PREV_ORDERS, _PROBE_CENTRES)
    phi_ep = EpisodeStats(
        num_drivers=1000,
        driver_capacity=4,
        speed_kmh=30.0,
        region_centres=_PROBE_CENTRES,
        region_neighbours=_PROBE_NEIGHBOURS,
        scale=6.0,
        reward_fn=reward_fn,
        objective_label="synthetic probe",
        fairness_strength=float(fairness_strength),
        od_count=od_count,
        od_out=od_out,
        od_in=od_in,
        od_orders=od_orders,
        dist=dist,
    )
    phi_step = GlobalStats(
        time=10.0,
        num_pending=20,
        num_drivers=1000,
        num_idle=800,
        total_free_capacity=3000,
        demand_pressure=0.5,
        mean_solo_time=6.0,
        region_demand=(3.0, 5.0, 1.0, 0.0),
        region_supply=(2.0, 1.0, 0.0, 1.0),
    )
    return phi_ep, phi_step


def _synthetic_inputs() -> Tuple[Dict, Dict, EpisodeStats, GlobalStats]:
    """A tiny but schema-faithful (driver_obs, order, phi_ep, phi_step) smoke input.

    Includes one onboard order so ``_onboard_slack`` / en-route logic exercises
    a non-trivial path, and a driver that can still fit the candidate order.
    """
    driver_obs = {
        "self": {
            "driver_id": 7,
            "location": (-73.98, 40.75),
            "status": "enroute",
            "capacity": 4,
            "speed": None,
            "onboard_passengers": 1,
            "assigned_orders": [100],
            "assigned_order_details": [
                {
                    "order_id": 100,
                    "origin": (-73.99, 40.75),
                    "destination": (-73.96, 40.78),
                    "num_passengers": 1,
                    "onboard": True,
                    "eta": 5.0,
                }
            ],
            "committed_passengers": 1,
            # A VALID index into the 4-region probe layout, computed by the same
            # partition rule the env uses. It used to be a hard-coded 42, which
            # meant a scorer indexing an array by ``current_region`` -- kappa,
            # od_out, region_neighbours -- blew up in the probe on code that is
            # fine on the real 100-region map, or slipped through here and blew
            # up mid-rollout.
            "current_region": _probe_region((-73.98, 40.75)),
        }
    }
    order = {
        "order_id": 42,
        "origin": (-73.97, 40.76),
        "destination": (-73.95, 40.79),
        "num_passengers": 1,
        "waiting_time": 1.0,
        # Both endpoints' regions, as the env now ships them on every pending order.
        "origin_region": _probe_region((-73.97, 40.76)),
        "destination_region": _probe_region((-73.95, 40.79)),
    }
    phi_ep, phi_step = _synthetic_phi()
    return driver_obs, order, phi_ep, phi_step


def validate_skill(skill: CompiledSkill) -> Tuple[bool, str]:
    """Smoke-run ``skill`` on synthetic inputs; return ``(ok, feedback)``.

    ``feedback`` is empty on success, else a message suitable both for the
    evolution log and for feeding back to the LLM.
    """
    driver_obs, order, phi_ep, phi_step = _synthetic_inputs()
    try:
        s = skill.score(driver_obs, order, phi_ep, phi_step)
    except Exception as e:  # noqa: BLE001 -- any runtime error is a rejection
        return False, f"score() raised on a normal input: {e!r}"
    if not isinstance(s, (int, float)) or isinstance(s, bool):
        return False, f"score() must return a number, got {type(s).__name__}."
    if not math.isfinite(float(s)):
        return False, f"score() returned a non-finite value: {s!r}."

    # Infeasible order (party exceeds free capacity) must be rejected, not crash.
    infeasible = dict(order, num_passengers=99)
    try:
        si = skill.score(driver_obs, infeasible, phi_ep, phi_step)
    except Exception as e:  # noqa: BLE001
        return False, f"score() raised on an infeasible (over-capacity) order: {e!r}"
    if not (isinstance(si, (int, float)) and math.isfinite(float(si))):
        return False, "score() must return a finite number for infeasible orders."

    try:
        n = skill.noop_score(driver_obs, phi_ep, phi_step)
    except Exception as e:  # noqa: BLE001
        return False, f"noop_score() raised: {e!r}"
    if not (isinstance(n, (int, float)) and math.isfinite(float(n))):
        return False, f"noop_score() must return a finite number, got {n!r}."

    return True, ""


def _synthetic_region_inputs(
    *, fairness_strength: float = 0.0
) -> Tuple[Dict, EpisodeStats, GlobalStats, RegionState]:
    """Schema-faithful (driver_obs, phi_ep, phi_step, kappa) for a reposition probe.

    Unlike :func:`_synthetic_inputs` (order scoring), this driver_obs carries the
    region machinery the reposition scorer reads: preset region centres, the env's
    region adjacency, pending-order pickups (the demand to chase), and the whole
    fleet's public state (free supply competing for the same demand). The driver is
    IDLE -- that is the only status that ever reaches the repositioner. ``kappa`` is
    seeded from ``phi_step`` exactly as :func:`choose_relocation_targets` does.

    The fairness multipliers are built from a synthetic income spread at the probed
    ``fairness_strength`` with the SAME :class:`~pref_dispatch.budget.FairnessBudget`
    the matcher uses, so probing at strength 0 hands the scorer an all-1.0 dict
    (fairness off) and probing high hands it a genuinely spread one. A scorer that
    divides by ``fairness_budget - 1`` is then caught here rather than mid-rollout.
    """
    # Driver 7 has earned below the fleet mean (so beta > 1 once the budget is on),
    # driver 8 above it -- the two cases a scorer must tell apart.
    _income = {7: 20.0, 8: 60.0}
    _betas = FairnessBudget(strength=fairness_strength).budgets(_income)
    driver_obs = {
        "self": {
            "driver_id": 7,
            "location": (-73.98, 40.75),
            "current_region": 0,
            "status": "idle",
            "capacity": 4,
            "committed_passengers": 0,
        },
        "relocation_points": _PROBE_CENTRES,
        "region_neighbours": _PROBE_NEIGHBOURS,
        "pending_orders": [
            # Region fields included exactly as the env ships them, computed by
            # the same partition rule, so a scorer may bucket pending demand by
            # ``origin_region`` / ``destination_region`` here as it does live.
            {"order_id": 200, "origin": (-73.95, 40.79),
             "destination": (-73.90, 40.80), "num_passengers": 2, "waiting_time": 3.0,
             "origin_region": _probe_region((-73.95, 40.79)),
             "destination_region": _probe_region((-73.90, 40.80))},
            {"order_id": 201, "origin": (-73.94, 40.80),
             "destination": (-73.99, 40.75), "num_passengers": 1, "waiting_time": 1.0,
             "origin_region": _probe_region((-73.94, 40.80)),
             "destination_region": _probe_region((-73.99, 40.75))},
        ],
        "all_drivers": {
            7: {"location": (-73.98, 40.75), "status": "idle", "onboard_passengers": 1},
            8: {"location": (-73.96, 40.78), "status": "idle", "onboard_passengers": 0},
        },
        "fairness_budget": _betas[7],
        "driver_budgets": _betas,
    }
    phi_ep, phi_step = _synthetic_phi(fairness_strength=fairness_strength)
    kappa = RegionState.from_phi_step(phi_step, len(_PROBE_CENTRES), supply_weight=1.0)
    return driver_obs, phi_ep, phi_step, kappa


# --------------------------------------------------------------------------- #
# Repositioner compilation + validation (Feature 3, Phase 2).                  #
# --------------------------------------------------------------------------- #
@dataclass
class CompiledRepositioner:
    """A validated per-region scorer:
    ``reposition_scores(driver_obs, phi_ep, phi_step, kappa, w)``.

    Returns ``{region_index: base_score}`` over a subset of the preset regions;
    higher = more worth cruising an idle empty car to. ``{}`` defers to the
    demand-gravity heuristic. ``kappa`` is the shared per-region demand/supply
    state and ``w`` is the episode objective (both readable by the scorer). This is
    the ONLY thing the LLM authors for the Feature-3 arm -- the deterministic
    coordinated-spreading, stay rules, and ``{"relocate": idx}`` emission stay in
    :mod:`pref_dispatch.reposition`, never model-authored.
    """

    name: str
    reposition_scores: Callable[
        [Dict, EpisodeStats, GlobalStats, RegionState, object], Dict[int, float]
    ]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"CompiledRepositioner({self.name})"


def compile_repositioner(code: str, name: str = "evolved") -> CompiledRepositioner:
    """Compile LLM repositioner ``code`` into a :class:`CompiledRepositioner`.

    ``code`` must define
    ``reposition_scores(driver_obs, phi_ep, phi_step, kappa, w) -> dict``. Same
    sandbox as skills but WITHOUT the order-scoring primitives (region scoring
    reasons over pending demand + fleet state + distances, not routing internals).
    Raises :class:`SandboxError`.
    """
    ns = _exec_restricted(code, {})
    fn = ns.get("reposition_scores")
    if not callable(fn):
        raise SandboxError(
            "repositioner code must define a callable `reposition_scores`."
        )
    return CompiledRepositioner(name=name, reposition_scores=fn)


def validate_repositioner(
    repositioner: CompiledRepositioner,
) -> Tuple[bool, str]:
    """Smoke-run ``reposition_scores`` on synthetic region inputs; ``(ok, feedback)``.

    Contract: returns a ``dict`` mapping region indices to finite numbers; ``{}``
    is allowed (defer to the demand-gravity heuristic). Every key must be an
    ``int`` region index in ``[0, n_regions)`` -- an out-of-range or non-int key
    would make the downstream ``{"relocate": idx}`` action illegal, so it is a
    hard rejection fed back to the LLM. Values must be finite (they seed the
    coordinated-spreading kernel, which cannot digest NaN). Probed both with an
    objective ``w`` present and with ``w=None`` so a scorer that dereferences ``w``
    unconditionally is caught, and at fairness strength 0 AND 2.5 so a scorer that
    branches on ``phi_ep.fairness_strength`` (or divides by it) is caught at both
    ends of the axis Phase 3 trains over.
    """
    driver_obs, phi_ep, phi_step, kappa = _synthetic_region_inputs()
    n_regions = len(driver_obs["relocation_points"])

    def _probe_w(event):  # a cheap stand-in objective the scorer may call
        return float(len(event.get("assigned_orders", [])))

    for w, strength in ((_probe_w, 0.0), (None, 0.0), (_probe_w, 2.5)):
        # Fresh inputs per probe: the scorer must not rely on mutation from a prior
        # call, and the fairness axis has to move (driver_obs carries the budgets,
        # phi_ep carries the strength) or a scorer reading it is never exercised.
        driver_obs, phi_ep, _, kappa = _synthetic_region_inputs(
            fairness_strength=strength)
        try:
            out = repositioner.reposition_scores(driver_obs, phi_ep, phi_step, kappa, w)
        except Exception as e:  # noqa: BLE001 -- any runtime error is a rejection
            return False, f"reposition_scores() raised on a normal input: {e!r}"
        if not isinstance(out, dict):
            return False, (
                f"reposition_scores() must return a dict, got {type(out).__name__}."
            )
        for k, v in out.items():
            if not isinstance(k, (int, np.integer)) or isinstance(k, bool):
                return False, (
                    f"reposition_scores() region key must be an int index, got "
                    f"{k!r} ({type(k).__name__})."
                )
            if not (0 <= int(k) < n_regions):
                return False, (
                    f"reposition_scores() returned region index {int(k)} out of range "
                    f"[0, {n_regions}); an out-of-range relocate target is illegal."
                )
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                return False, (
                    f"reposition_scores() value for region {int(k)} must be a number, "
                    f"got {type(v).__name__}."
                )
            if not math.isfinite(float(v)):
                return False, (
                    f"reposition_scores() value for region {int(k)} is non-finite: {v!r}."
                )
    return True, ""


# Three probe dicts a fitness is smoke-run on. The numbers are MEASURED, not
# invented: both non-empty dicts are real ``finalize()`` output from one full
# Mon-18h hour (capacity 7, 40 km/h) driven by the handwritten ``revenue`` skill,
# at the two ends of the fleet range Phase 1 searches over.
#
# Getting these magnitudes right is not cosmetic. The old single probe carried
# ``detour_total = 143.2`` where a real hour produces 10,000-40,000, i.e. it was
# ~300x too small, so a fitness whose terms are mixed in the WRONG UNITS looked
# perfectly healthy here and was catastrophic on real metrics. The case that cost
# a whole search: ``completed + 40*service_rate - 0.5*detour_total`` scored +681.6
# on the old probe and about -29,600 on a real hour -- which made "serve nobody"
# (every term 0) that fitness's global optimum, and the search found it.
_PROBE_BUSY: Dict[str, float] = {
    # 1829 cars, 18:00 Monday: supply-rich, nearly everything gets served.
    "revenue": 107366.2,      # sum of solo_time(min) x party over assigned orders
    "service_rate": 0.982,    # assigned / total_orders in [0, 1]
    "completed": 6225,        # orders delivered
    "assigned": 8430,         # orders assigned
    "mean_service_time": 11.76,   # mean end-to-end service time (min)
    "detour_total": 37940.6,      # total extra detour time (min)
    "income_gini": 0.156,     # driver-income Gini in [0, 1]
    "income_cv": 0.938,       # driver-income coefficient of variation
    "income_mean": 2.309,     # mean per-driver cumulative reward
    "income_min": -5.488,     # worst-off driver -- NEGATIVE is normal, not a bug
}
_PROBE_SCARCE: Dict[str, float] = {
    # The same hour with 200 cars: the scarcity end, where most orders go unserved.
    "revenue": 36390.0,
    "service_rate": 0.205,
    "completed": 1179,
    "assigned": 1760,
    "mean_service_time": 17.52,
    "detour_total": 10798.6,
    "income_gini": 0.168,
    "income_cv": 0.769,
    "income_mean": 3.101,
    "income_min": -4.835,
}
_PROBE_IDLE: Dict[str, float] = {
    # Nobody served at all. Not a hypothetical: two frozen skills evolved into
    # exactly this because it was their authored fitness's best point. Probing it
    # only checks the fitness does not RAISE or go non-finite here (a 0/0 or a
    # log(0) is a bug); whether do-nothing scores well is a question for the
    # post-search audit (pref_dispatch/llm/skill_audit.py), never a hard rule.
    "revenue": 0.0,
    "service_rate": 0.0,
    "completed": 0,
    "assigned": 0,
    "mean_service_time": 0.0,
    "detour_total": 0.0,
    "income_gini": 0.0,
    "income_cv": 0.0,
    "income_mean": 0.0,
    "income_min": 0.0,
}


def _synthetic_metrics() -> Dict[str, float]:
    """One representative episode-summary dict (the busy end).

    Keys and units MUST match exactly what
    :meth:`pref_dispatch.metrics.EpisodeMetrics.finalize` returns, so a fitness
    that validates here also runs on real rollout metrics.
    """
    return dict(_PROBE_BUSY)


def fitness_probes() -> Dict[str, Dict[str, float]]:
    """The probe dicts ``validate_fitness`` runs, by name (for prompts/tests)."""
    return {"busy_hour": dict(_PROBE_BUSY),
            "scarce_hour": dict(_PROBE_SCARCE),
            "served_nobody": dict(_PROBE_IDLE)}


def validate_fitness(fn: Callable[[Dict], float]) -> Tuple[bool, str]:
    """Smoke-run a fitness fn on all three probe dicts; ``(ok, feedback)``.

    Three, not one, because the failures worth catching before a search starts are
    exactly the ones a single mid-range dict hides: a term that only blows up at
    scarcity, and a division that only blows up when nothing was served.
    """
    for label, probe in fitness_probes().items():
        try:
            v = fn(dict(probe))
        except Exception as e:  # noqa: BLE001
            return False, (f"fitness() raised on the {label} metrics dict "
                           f"({probe}): {e!r}")
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return False, (f"fitness() must return a number, got "
                           f"{type(v).__name__} on the {label} dict.")
        if not math.isfinite(float(v)):
            return False, (f"fitness() returned a non-finite value on the "
                           f"{label} dict: {v!r}.")
    return True, ""


def normalise_terms(
    fn: Callable[[Dict], float],
    probes: Sequence[Dict[str, float]],
    *,
    domain: str,
) -> Tuple[Callable[[Dict], float], float]:
    """Rescale a self-authored scalar so its term weights sum to ONE.

    The LLM authors the coefficients of its own objective freely (a Phase-1
    ``fitness(metrics)`` or a Phase-2/3 ``reward(event)``), so an extreme term can
    carry a scale that dwarfs the others -- and a probe-based reader (the combiner
    probing ``w``, or the group-relative fitness) would read that scale as the term
    being "dominant" even when the author meant a mild tilt. Normalisation divides
    the whole scalar by ``S`` = the sum of its absolute values on the supplied
    ``probes``, making the term weights sum to one while preserving their RATIOS.

    ``domain`` is a label for provenance/logging ("reward" or "fitness"). The
    scaling is best-effort: an unreadable or degenerate fn (non-finite ``S`` or
    ``S <= 1e-12``) is returned unchanged with ``S = 1.0``.
    """
    del domain  # provenance label, kept for call-sites' logging only
    try:
        total = sum(abs(float(fn(dict(p)))) for p in probes)
    except Exception:  # noqa: BLE001 -- normalisation must never reject a program
        return fn, 1.0
    if not math.isfinite(total) or total <= 1e-12:
        return fn, 1.0
    S = float(total)

    def scaled(inputs: Dict) -> float:
        return float(fn(inputs)) / S

    return scaled, S


# --------------------------------------------------------------------------- #
# Reward compilation + validation (LLM-authored platform reward, §Phase-2).    #
# --------------------------------------------------------------------------- #
def compile_reward(code: str) -> Callable[[Dict], float]:
    """Compile an LLM self-authored ``reward(event) -> float``.

    ``event`` is the env's per-driver, per-step event dict (the 11 keys of
    :meth:`ride_gym.env.RidePoolEnv._new_event`). Like fitness, no primitives and
    no env are injected -- only ``math`` / ``np`` and safe builtins over the event
    dict -- so the body can only do cheap arithmetic over what it is handed, which
    is exactly the ``RewardFunction`` call surface the env grades with. Raises
    :class:`SandboxError`.
    """
    ns = _exec_restricted(code, {})
    fn = ns.get("reward")
    if not callable(fn):
        raise SandboxError("reward code must define a callable `reward`.")
    return fn


def _synthetic_events() -> Tuple[Dict, Dict]:
    """Return (populated, empty) per-step event dicts for a reward smoke run.

    Keys and units MUST match exactly what
    :meth:`ride_gym.env.RidePoolEnv._new_event` creates and ``_assign_orders`` /
    ``_move_driver`` populate, so a reward that validates here also runs on the
    real per-step events the env grades with. The populated variant exercises the
    assignment / revenue / service-time / detour paths (non-empty sub-dicts, a
    signed ``extra_detour_time``); the empty variant is an idle-wait no-op step.
    """
    populated = {
        "assigned_orders": [100, 101],
        "assigned_party_sizes": {100: 1, 101: 3},
        "assigned_solo_times": {100: 6.5, 101: 4.0},
        "assigned_service_times": {100: 11.2, 101: 7.8},
        "assigned_dispatch_wait": {100: 2.0, 101: 1.5},
        "assigned_pickup_times": {100: 3.2, 101: 2.6},
        "assigned_detour_times": {100: 1.5, 101: 1.2},
        "completed_orders": [77],
        "picked_up_orders": [100],
        "distance_moved": 0.42,
        "time_moved": 1.5,
        "is_empty_move": False,
        "is_idle_wait": False,
        "extra_detour_time": 2.3,
    }
    empty = {
        "assigned_orders": [],
        "assigned_party_sizes": {},
        "assigned_solo_times": {},
        "assigned_service_times": {},
        "assigned_dispatch_wait": {},
        "assigned_pickup_times": {},
        "assigned_detour_times": {},
        "completed_orders": [],
        "picked_up_orders": [],
        "distance_moved": 0.0,
        "time_moved": 0.0,
        "is_empty_move": False,
        "is_idle_wait": True,
        "extra_detour_time": 0.0,
    }
    return populated, empty


def validate_reward(fn: Callable[[Dict], float]) -> Tuple[bool, str]:
    """Smoke-run a reward fn on synthetic event dicts; return ``(ok, feedback)``.

    Runs both a populated (assignment/detour) event and an empty (idle) event so a
    reward that crashes only on empty sub-dicts is caught. ``feedback`` is empty on
    success, else a message suitable both for the log and for feeding back to the
    LLM.
    """
    populated, empty = _synthetic_events()
    for label, ev in (("a populated", populated), ("an empty", empty)):
        try:
            v = fn(ev)
        except Exception as e:  # noqa: BLE001 -- any runtime error is a rejection
            return False, f"reward() raised on {label} event dict: {e!r}"
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return False, (
                f"reward() must return a number on {label} event, "
                f"got {type(v).__name__}."
            )
        if not math.isfinite(float(v)):
            return False, f"reward() returned a non-finite value on {label} event: {v!r}."
    return True, ""


# --------------------------------------------------------------------------- #
# Combiner compilation + validation (Phase 2 upper layer, §5).                 #
# --------------------------------------------------------------------------- #
@dataclass
class CompiledCombiner:
    """A validated upper-layer scorer:
    ``skill_scores(driver_obs, phi_ep, phi_step, w)``.

    Returns a dict mapping (a subset of) frozen skill names to real scores. The
    adapter (:class:`pref_dispatch.llm.combiner_adapter.LLMCombiner`) turns those
    scores into the one-hot weight dict the matcher consumes -- the LLM never
    touches the frozen skills' code, it only *scores* them (§5.1 boundary). ``w``
    is the episode objective (a callable reward fn, or ``None``); the combiner MAY
    call it to self-derive the blend (the ``reward_aware_dispatcher_v2`` pattern).
    """

    skill_scores: Callable[
        [Dict, EpisodeStats, GlobalStats, object], Dict[str, float]
    ]


def compile_combiner(code: str) -> CompiledCombiner:
    """Compile LLM combiner ``code`` into a :class:`CompiledCombiner`.

    ``code`` must define ``skill_scores(driver_obs, phi_ep, phi_step, w) -> dict``.
    Same sandbox as skills but WITHOUT the skill primitives (the combiner reasons
    over the driver state + two-layer stats + objective, not routing internals).
    Raises :class:`SandboxError`.
    """
    ns = _exec_restricted(code, {})
    fn = ns.get("skill_scores")
    if not callable(fn):
        raise SandboxError("combiner code must define a callable `skill_scores`.")
    return CompiledCombiner(skill_scores=fn)


def validate_combiner(
    combiner: CompiledCombiner, skill_names: Tuple[str, ...]
) -> Tuple[bool, str]:
    """Smoke-run ``skill_scores`` on synthetic inputs; return ``(ok, feedback)``.

    Checks it returns a dict of finite numbers keyed by *known* frozen skill
    names (an out-of-basis key is a contract violation, fed back to the LLM), and
    that at least one known skill is scored (else argmax is undefined). Probes both
    an objective ``w`` present and ``w=None`` so a combiner that ignores ``w`` still
    validates but a crash on either is caught.
    """
    driver_obs, _order, phi_ep, phi_step = _synthetic_inputs()

    def _probe_w(event):  # a cheap stand-in objective the combiner may call
        return float(len(event.get("assigned_orders", [])))

    known = set(skill_names)
    for w in (_probe_w, None):
        try:
            out = combiner.skill_scores(driver_obs, phi_ep, phi_step, w)
        except Exception as e:  # noqa: BLE001 -- any runtime error is a rejection
            return False, f"skill_scores() raised on a normal input: {e!r}"
        if not isinstance(out, dict):
            return False, (
                f"skill_scores() must return a dict, got {type(out).__name__}."
            )
        scored_known = [k for k in out if k in known]
        if not scored_known:
            return False, (
                "skill_scores() scored none of the frozen skills. It must return "
                f"scores for skills from: {sorted(known)}."
            )
        for k, v in out.items():
            if k not in known:
                return False, (
                    f"skill_scores() returned an unknown skill name {k!r}; only "
                    f"these frozen skills exist: {sorted(known)}."
                )
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                return False, f"score for {k!r} must be a number, got {type(v).__name__}."
            if not math.isfinite(float(v)):
                return False, f"score for {k!r} is non-finite: {v!r}."
    return True, ""


# --------------------------------------------------------------------------- #
# Convenience: compile+validate with graceful fallback to a seed.             #
# --------------------------------------------------------------------------- #
def safe_compile_skill(
    code: str,
    name: str = "evolved",
    fallback: Optional[CompiledSkill] = None,
) -> Tuple[Optional[CompiledSkill], str]:
    """Compile + validate ``code``; on any failure return ``(fallback, why)``.

    The evolution loop uses ``why`` both to log the rejection and to feed the
    error back to the LLM on the next generation. If ``fallback`` is ``None`` the
    first element is ``None`` (caller drops the candidate).
    """
    try:
        skill = compile_skill(code, name=name)
    except SandboxError as e:
        return fallback, str(e)
    ok, feedback = validate_skill(skill)
    if not ok:
        return fallback, feedback
    return skill, ""


def safe_compile_repositioner(
    code: str,
    name: str = "evolved",
    fallback: Optional[CompiledRepositioner] = None,
) -> Tuple[Optional[CompiledRepositioner], str]:
    """Compile + validate repositioner ``code``; on any failure ``(fallback, why)``.

    Same shape as :func:`safe_compile_skill` but for the independent
    Feature-3 reposition scorer path.
    """
    try:
        repositioner = compile_repositioner(code, name=name)
    except SandboxError as e:
        return fallback, str(e)
    ok, feedback = validate_repositioner(repositioner)
    if not ok:
        return fallback, feedback
    return repositioner, ""
