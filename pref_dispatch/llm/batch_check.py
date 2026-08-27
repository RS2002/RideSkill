"""Batch-coverage software checks for the per-generation training batch.

Each phase draws a training batch per generation:
  * Phase 1: a batch of SCENARIOS (one skill, one fixed objective).
  * Phase 2: a batch of (scenario, objective) cells.
  * Phase 3: a batch of (scenario, objective, fairness-strength) cells.

The samplers already enforce MECHANICAL stratification (fleet bands, objective
family minimums, strength bands). This module adds the checks the samplers
cannot express: does the batch actually SPAN the metric axes the objective
functions price, and the scene dimensions that change the operating point? It is
all SOFTWARE -- ``w`` is called on term-isolating probes and the differences are
read directly, no LLM in the loop -- so a check can never hang the run. The LLM
self-check (batch diversity prompt) is wired separately and non-blocking.

The scene half is structural (fleet band / regime / high-low volume), the
objective half is semantic (which event terms ``w`` prices). Both degrade
GRACEFULLY: a batch that fails is reported, and only a caller that opts in
passes ``reject=True`` to make a failure a hard error.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# Metrc-axis isolation probes, keyed by axis name. Each baseline/pair isolates
# ONE event term so that ``w(high) - w(low)`` is (a scaled estimate of) how much
# the objective prices that axis. ``None`` for w means the objective is blind --
# every axis scores 0, which is reported as the blind family.
#
# Event keys must match the env's per-step event dict exactly
# (ride_gym.env._new_event / _assign_orders / _move_driver).
def _ev(**kw) -> Dict:
    """A populated per-step event with the given overrides."""
    base = {
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
        "is_idle_wait": False,
        "extra_detour_time": 0.0,
    }
    base.update(kw)
    return base


# Each entry: (axis, [(label, event_high, event_low)]).
def objective_axis_probes() -> Dict[str, List[Tuple[str, Dict, Dict]]]:
    """Term-isolating event pairs per objective axis.

    Every probe compares a HIGH event to a LOW event that differ ONLY in the axis
    being priced (the baseline has a single assigned order at a reference state).
    An objective that prices a term returns a nonzero (high - low) difference on
    its axis; an objective that ignores it returns ~0 on that axis. The reference
    state -- one order, no onboard, base solo/service/pickup/wait -- is shared so
    the differences are comparable across objectives in a batch.
    """
    p = 101                        # the single assigned order in every probe
    BASE = dict(
        assigned_orders=[p],
        assigned_party_sizes={p: 1},
        assigned_solo_times={p: 6.0},
        assigned_service_times={p: 11.0},
        assigned_dispatch_wait={p: 2.0},
        assigned_pickup_times={p: 3.0},
        assigned_detour_times={p: 0.0},
        extra_detour_time=0.0,
        completed_orders=[],
        picked_up_orders=[],
        distance_moved=0.0,
        time_moved=0.0,
        is_empty_move=False,
        is_idle_wait=False,
    )

    def event(**ow) -> Dict:
        d = dict(BASE)
        d.update(ow)
        return d

    completion = [
        ("dropped_off", event(completed_orders=[p]), event(completed_orders=[])),
    ]
    seating = [
        ("party2_vs_1", event(assigned_party_sizes={p: 2}),
         event(assigned_party_sizes={p: 1})),
    ]
    volume = [
        ("two_vs_one", event(assigned_orders=[p, 202],
                             assigned_party_sizes={p: 1, 202: 1},
                             assigned_solo_times={p: 6.0, 202: 6.0},
                             assigned_service_times={p: 11.0, 202: 11.0},
                             assigned_dispatch_wait={p: 2.0, 202: 2.0},
                             assigned_pickup_times={p: 3.0, 202: 3.0}),
         event(assigned_orders=[p])),
    ]
    dispatch_wait = [
        ("wait8_vs_2", event(assigned_dispatch_wait={p: 8.0}),
         event(assigned_dispatch_wait={p: 2.0})),
    ]
    pickup = [
        ("pickup8_vs_3", event(assigned_pickup_times={p: 8.0}),
         event(assigned_pickup_times={p: 3.0})),
    ]
    service = [
        ("svc18_vs_11", event(assigned_service_times={p: 18.0}),
         event(assigned_service_times={p: 11.0})),
    ]
    solo_len = [
        ("solo12_vs_6", event(assigned_solo_times={p: 12.0},
                              assigned_service_times={p: 17.0}),
         event(assigned_solo_times={p: 6.0},
               assigned_service_times={p: 11.0})),
    ]
    detour = [
        ("det4_vs_0", event(assigned_detour_times={p: 4.0},
                            extra_detour_time=0.0),
         event(assigned_detour_times={p: 0.0}, extra_detour_time=0.0)),
    ]
    detour_onboard = [
        ("ob_det4_vs_0", event(assigned_detour_times={p: 0.0},
                               extra_detour_time=4.0),
         event(assigned_detour_times={p: 0.0}, extra_detour_time=0.0)),
    ]
    empty = [
        ("empty_vs_stay", event(is_empty_move=True, time_moved=1.0,
                                distance_moved=1.0),
         event(is_empty_move=False, time_moved=0.0, distance_moved=0.0)),
    ]
    idle = [
        ("idle_vs_wait", event(is_idle_wait=True, time_moved=1.0),
         event(is_idle_wait=False, time_moved=0.0)),
    ]
    revenue = [
        ("solo_long_vs_short", event(assigned_solo_times={p: 12.0},
                                     assigned_party_sizes={p: 1}),
         event(assigned_solo_times={p: 6.0}, assigned_party_sizes={p: 1})),
    ]
    return {
        "completion": completion,
        "seating": seating,
        "volume": volume,
        "dispatch_wait": dispatch_wait,
        "pickup": pickup,
        "service": service,
        "solo_len": solo_len,
        "detour": detour,
        "detour_onboard": detour_onboard,
        "empty": empty,
        "idle": idle,
        "revenue": revenue,
    }


# The metric axes that matter for the LLM prompt (the ones the user asked the
# combiner to read). A batch whose objectives together price FEW of these leaves
# the metric dimension of generality unexamined. Detour is split into the
# NEW-order per-order pooled detour and the ONBOARD re-routing impact, because a
# reward may price one without the other.
KEY_AXES: Tuple[str, ...] = (
    "completion", "seating", "volume", "dispatch_wait", "pickup", "service",
    "solo_len", "detour", "detour_onboard", "empty", "idle", "revenue",
)


def _axis_diff(w: Callable[[Dict], float], high: Dict, low: Dict) -> float:
    try:
        return float(w(dict(high))) - float(w(dict(low)))
    except Exception:
        return 0.0   # a probe that breaks is an axis the objective does not price


def objective_axis_profile(
    w: Optional[Callable[[Dict], float]],
) -> Dict[str, float]:
    """Per-axis pricing magnitude of one objective, by term-isolating probes.

    Returns ``{axis: |w(high) - w(low)|}`` -- the scale on which the objective
    prices that axis. A None ``w`` scores every axis 0 (the blind objective).

    v2 (2026-08-26): NO normalisation against the objective's own largest term.
    The old path divided each axis by ``maxdiff`` so the profile compared SHAPE
    not scale -- but that is a normalisation on the w RESPONSES, which the user
    flagged as a second, unnecessary scale on top of the reward's own
    coefficient scale. The batch check now reports the raw probe deltas; the
    coefficient normalisation (if any) happens once, at reward authoring time,
    and is what the axes should read through.
    """
    profile = {ax: 0.0 for ax in KEY_AXES}
    if w is None:
        return profile
    for ax, pairs in objective_axis_probes().items():
        profile[ax] = max(abs(_axis_diff(w, h, lo)) for _, h, lo in pairs)
    return profile


def objective_batch_coverage(
    objectives: Sequence[object],
) -> Dict[str, object]:
    """Which axis dimensions the objective batch CO-LOCALLY prices.

    An axis is "covered" if at least one objective prices it at >= coverage_thresh
    of its own dominant term. Returns ``{axes, per_axis, n_axes_covered,
    covered_axes, missing_axes, family_mix}`` for the batch.
    """
    per_axis: Dict[str, float] = {ax: 0.0 for ax in KEY_AXES}
    family_counter: Counter = Counter()
    n_blind = 0

    for ob in objectives:
        w = getattr(ob, "w", None)
        if getattr(ob, "reward_function", None) is None or w is None:
            n_blind += 1
            family_counter[getattr(ob, "family", "?")] += 1
            continue
        prof = objective_axis_profile(w)
        family_counter[getattr(ob, "family", "?")] += 1
        # An objective COVERS the axes it prices. Without a w-response scale we
        # cannot compare two objectives' raw deltas, so we classify WITHIN one
        # objective: the axis it prices most heavily is its dominant term, and any
        # axis within 50% of that dominant is also "priced" by it.
        peak = max(prof.values(), default=0.0)
        if peak <= 0.0:
            continue
        for ax, val in prof.items():
            # val >= 0.5 * peak: this axis is priced at >= half the objective's
            # biggest term. Raw delta, no cross-objective normalisation.
            if val >= 0.5 * peak:
                per_axis[ax] = max(per_axis[ax], val)

    covered = [ax for ax, val in per_axis.items() if val > 0.0]
    missing = [ax for ax in KEY_AXES if ax not in covered]
    return {
        "per_axis": per_axis,
        "n_axes_covered": len(covered),
        "covered_axes": covered,
        "missing_axes": missing,
        "family_mix": dict(family_counter),
        "n_blind": n_blind,
    }


def scene_band_counts(
    scenarios: Sequence[object],
    bands: Sequence[Tuple[int, int]],
) -> Dict[str, int]:
    """Fleet-band histogram of a scene batch (band label -> count)."""
    from pref_dispatch.llm.batch_pairing import band_index  # local to avoid cycle

    counts: Counter = Counter()
    for sc in scenarios:
        bi = band_index(sc, bands)
        lo, hi = bands[bi]
        counts[f"fleet{lo}-{hi}"] += 1
    return dict(counts)


def scene_regime_counts(scenarios: Sequence[object]) -> Dict[str, int]:
    """Time-of-day regime histogram (offpeak / shoulder / peak)."""
    counts: Counter = Counter()
    for sc in scenarios:
        counts[getattr(sc, "regime", "?")] += 1
    return dict(counts)


def scene_coverage_report(
    scenarios: Sequence[object],
    bands: Sequence[Tuple[int, int]],
    *,
    min_bands: int = 2,
    min_regimes: int = 2,
    require_high_volume: bool = False,
) -> Dict[str, object]:
    """Structural scene coverage: does the batch span scale + time-of-day.

    This is the SOFTWARE counterpart of the LLM scene-diversity self-check. It
    cannot see "how busy" a window is from the label, so it uses the fleet band
    (scale) and the regime (time-of-day) as the two independent axes that change
    the operating point, and -- when the sampler guarantees at least one busy hour
    -- the count of distinct windows as a proxy for demand-shape variety.
    """
    bands_c = scene_band_counts(scenarios, bands)
    regimes_c = scene_regime_counts(scenarios)
    windows = {getattr(sc, "window", "?") for sc in scenarios}
    ok_bands = len(bands_c) >= min_bands
    ok_regimes = len(regimes_c) >= min_regimes
    ok_windows = len(windows) >= 2
    return {
        "bands": bands_c,
        "regimes": regimes_c,
        "n_distinct_windows": len(windows),
        "ok_bands": ok_bands,
        "ok_regimes": ok_regimes,
        "ok_windows": ok_windows,
        "ok": ok_bands and ok_regimes and ok_windows,
    }


def batch_coverage_summary(
    scenarios: Sequence[object],
    objectives: Sequence[object],
    bands: Sequence[Tuple[int, int]],
) -> Dict[str, object]:
    """One call combining the objective + scene coverage checks.

    Pure function; no sampling, no LLM. Callers use it to (a) log the per-round
    coverage (so a run is auditable), (b) optionally reject a round that is too
    narrow, and (c) feed the LLM batch-diversity prompt a concrete summary to
    reason over, instead of leaving it to guess from labels.
    """
    sys_errors = []
    obj_cov = objective_batch_coverage(objectives)
    scen_cov = scene_coverage_report(scenarios, bands)
    if not scen_cov["ok"]:
        sys_errors.append(
            f"scene batch narrow: bands={scen_cov['bands']} "
            f"regimes={scen_cov['regimes']} "
            f"windows={scen_cov['n_distinct_windows']}"
        )
    if obj_cov["n_axes_covered"] < len(KEY_AXES) // 2:
        sys_errors.append(
            f"objective batch narrow: covers only {obj_cov['n_axes_covered']} "
            f"axes of {len(KEY_AXES)} ({obj_cov['covered_axes']})"
        )
    return {
        "objective": obj_cov,
        "scene": scen_cov,
        "sys_errors": sys_errors,
    }


# --------------------------------------------------------------------------- #
# LLM batch-diversity prompt (advisory, NON-blocking)                          #
# --------------------------------------------------------------------------- #
# The software check above can only count which axes OBJECTIVES price and which
# BANDS/regimes scenes span -- it cannot judge semantics ("these two objectives
# are the same niche reworded", "this batch never exercises scarcity"). That
# semantic judgement is what the per-round LLM self-check asks for. It is
# DELIBERATELY advisory: a failing answer is logged, never rejected, so an
# unparseable or evasive reply costs one call and the run continues. The
# software checks are the gate; the LLM check is the interpretation.
def build_batch_diversity_prompt(
    round_idx: int,
    *,
    phase: str,
    coverage: Dict[str, object],
    objectives: Sequence[object],
    scenarios: Sequence[object],
) -> Dict[str, str]:
    obj_cov = coverage.get("objective", {})
    scen_cov = coverage.get("scene", {})
    obj_lines = [
        f"  family={getattr(o, 'family', '?')} label={getattr(o, 'label', '?')!r} "
        f"spec={str(getattr(o, 'spec_text', ''))[:60]!r}"
        for o in objectives
    ]
    scen_lines = [
        f"  fleet={getattr(sc, 'num_drivers', '?')} regime={getattr(sc, 'regime', '?')} "
        f"window={getattr(sc, 'window', '?')}"
        for sc in scenarios
    ]
    user = (
        f"[phase {phase}] round {round_idx}: check this training batch's COVERAGE.\n\n"
        f"SCENARIOS ({len(scen_lines)}):\n" + "\n".join(scen_lines) + "\n\n"
        f"OBJECTIVES ({len(obj_lines)}):\n" + "\n".join(obj_lines) + "\n\n"
        f"MEASURED OBJECTIVE AXES (priced by w, from software probes):\n"
        f"  covered={obj_cov.get('covered_axes')}\n"
        f"  missing={obj_cov.get('missing_axes')}\n"
        f"  families={obj_cov.get('family_mix')}\n\n"
        f"MEASURED SCENE SPAN:\n"
        f"  bands={scen_cov.get('bands')} regimes={scen_cov.get('regimes')} "
        f"windows={scen_cov.get('n_distinct_windows')}\n\n"
        "Answer in 2-3 sentences: is this batch BROAD ENOUGH to train a program "
        "that generalises across objective term-shapes AND across fleet scale / "
        "time-of-day? If a dimension is under-covered (an axis no objective prices, "
        "a too-narrow scene span, or two objectives that are the same niche in new "
        "words), say exactly what is missing. Do NOT propose new code -- this is a "
        "coverage audit, not a generation call. End with 'COVERAGE_OK' or "
        "'COVERAGE_GAP: <what is missing>'."
    )
    return {
        "system": (
            "You audit ride-pooling training batches for coverage. You judge "
            "whether a batch of training scenarios and objectives spans the "
            "dimensions that determine how a learned program generalises. Answer "
            "shortly and factually; never invent objectives or scenes."
        ),
        "user": user,
    }
