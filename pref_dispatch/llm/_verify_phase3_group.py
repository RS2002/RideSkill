"""Offline, key-free end-to-end check of the Phase-3 group-relative loop (B3e).

Runs :func:`pref_dispatch.llm.evolve_reposition.evolve_repositioner_group` against
a scripted client and stubbed rollouts, so nothing touches the network or the real
environment. Phase 3 is Phase 2's search with ONE extra axis -- the fairness
strength -- so the checks are Phase 2's plus that axis:

* the round's cells spread every objective family over BOTH the fleet bands and the
  fairness bands (``pair_by_strength_band``), and the two bandings stay uncorrelated;
* a cell's group holds the two fixed anchors (the demand-gravity heuristic and
  repositioning switched OFF), so "better than everyone alive this round" is still
  negative when everyone alive is worse than not repositioning;
* fitness is standardised WITHIN a cell (reward scale cannot matter);
* selection key is the PURE GRPO mean advantage (``beta = 0``, 2026-08-13), and an
  elite slot is reserved per family and per strength band;
* behavioural clones are eliminated;
* a scorer that breaks at runtime gets one repair and is eliminated if it still breaks;
* crossover, mutation and parentless-fresh operators all get reached;
* the per-round checkpoint fires once per round, on rotating cells;
* the fixed-batch yardstick counts wins against BOTH anchors.

Run: ``python -m pref_dispatch.llm._verify_phase3_group``
"""

from __future__ import annotations

import json
import random
from typing import Dict, List, Optional, Sequence

from pref_dispatch.llm import evolve_reposition as E
from pref_dispatch.llm.batch_pairing import (
    DEFAULT_FLEET_BANDS,
    band_index,
    pair_by_fleet_band,
    pair_by_strength_band,
)
from pref_dispatch.llm.reposition_adapter import GuardedScorer
from pref_dispatch.llm.reposition_eval import (
    RepositionEval,
    group_evals,
    strength_label,
)
from pref_dispatch.llm.sandbox import compile_repositioner
from pref_dispatch.scenario import Scenario

# --------------------------------------------------------------------------- #
# Fixtures: objectives, cells, a scripted client, a stubbed rollout.          #
# --------------------------------------------------------------------------- #


class _Obj:
    """The two fields the group scorer reads off an objective: family and ``w``."""

    def __init__(self, family: str, w=None) -> None:
        self.family = family
        self.w = w


def _code(k: float) -> str:
    """A valid, distinct ``reposition_scores``. Distinctness is cosmetic here --
    the stubbed rollout keys rewards off the candidate NAME -- but two candidates
    sharing a program would make the clone check meaningless."""
    return (
        "def reposition_scores(driver_obs, phi_ep, phi_step, kappa, w):\n"
        f"    return {{0: {k}}}\n"
    )


class ScriptedClient:
    """Returns a fresh valid scorer per call; records every prompt it was sent."""

    def __init__(self) -> None:
        self.prompts: List[str] = []
        self.n = 0

    def complete(self, system: str, user: str, temperature=None) -> str:
        self.prompts.append(user)
        self.n += 1
        return json.dumps({
            "reposition_understanding":
                "Idle cars are worth moving only toward demand they can still reach "
                "before someone else does.",
            "skill_name": f"v{self.n}",
            "objective": "Send idle cars to under-supplied nearby regions.",
            "description": "Ranks regions by unmet demand over cruise time, and "
                           "returns nothing when no region clears the bar.",
            "code": _code(float(self.n)),
        })


_SCENARIOS = [
    Scenario(num_drivers=300, driver_capacity=3, speed_kmh=20.0, regime="peak", seed=1),
    Scenario(num_drivers=300, driver_capacity=3, speed_kmh=20.0, regime="offpeak", seed=2),
    Scenario(num_drivers=1200, driver_capacity=3, speed_kmh=20.0, regime="peak", seed=3),
    Scenario(num_drivers=1200, driver_capacity=3, speed_kmh=20.0, regime="offpeak", seed=4),
]
_OBJECTIVES = [_Obj("raw"), _Obj("completion"), _Obj("raw"), _Obj("pooling")]
_STRENGTHS = [0.0, 0.6, 2.0, 0.0]


class RolloutStub:
    """Deterministic fake rollout: reward depends on the scorer AND the cell.

    Cell levels differ by 1000x on purpose -- if the loop averaged raw rewards
    instead of standardising per cell, the big cell would decide everything and
    the assertions below would fail.
    """

    LEVELS = [10.0, 10_000.0, 3.0, 30_000.0]

    def __init__(self, broken: Sequence[str] = ()) -> None:
        self.broken = set(broken)
        self.calls = 0
        self.clone = False       # every scorer produces the SAME row
        # First-seen order gives each scorer its own reward level. Keying off the
        # NAME (rather than off nothing) matters: identical rows are what the loop
        # calls a behavioural clone and kills, so a stub that returned the same row
        # for everyone would silently reduce every round to one survivor.
        self._level: Dict[str, int] = {}

    def __call__(self, scorer, combiner, skills, scs, objs, sts, capture=400):
        self.calls += 1
        name = getattr(getattr(scorer, "scorer", None), "name", "?")
        if name in self.broken or ("*" in self.broken):
            # Telemetry is how the loop learns a program broke: the guard counts
            # the decisions it could not serve, it is not inferred from the row.
            scorer.n_calls = 100
            scorer.n_fallbacks = 100
        else:
            scorer.n_calls = 100
            scorer.n_defers = 5
        mult = 1.0 if self.clone else 1.3 + 0.1 * self._level.setdefault(
            name, len(self._level))
        return [RolloutStub.LEVELS[i % len(RolloutStub.LEVELS)] * mult
                for i in range(len(scs))]


def _anchor_stub(combiner, skills, scs, objs, sts):
    """(demand-gravity heuristic, reposition OFF) per cell, on the cell's scale."""
    return [(RolloutStub.LEVELS[i % len(RolloutStub.LEVELS)] * 1.05,
             RolloutStub.LEVELS[i % len(RolloutStub.LEVELS)] * 0.95)
            for i in range(len(scs))]


def _check(label: str, ok: bool, detail: str = "") -> None:
    if not ok:
        raise AssertionError(f"{label} FAILED {detail}")
    print(f"[B3] {label} OK {detail}".rstrip())


def _mk_eval(fitness: float, per_family: Dict[str, float],
             per_strength: Dict[str, float]) -> RepositionEval:
    return RepositionEval(fitness=fitness, raw_fitness=fitness, fallback_rate=0.0,
                          per_family=dict(per_family),
                          per_strength=dict(per_strength))


def _fake_candidate(name: str, fitness: float, per_family: Dict[str, float],
                    per_strength: Dict[str, float]) -> E.RepositionerCandidate:
    return E.RepositionerCandidate(
        meta={"skill_name": name, "code": _code(1.0)},
        scorer=compile_repositioner(_code(1.0), name=name),
        evaluation=_mk_eval(fitness, per_family, per_strength),
    )


# --------------------------------------------------------------------------- #
# Checks.                                                                     #
# --------------------------------------------------------------------------- #

def check_strength_pairing() -> None:
    """Every family is measured with the budget OFF and ON, at every fleet scale."""
    rng = random.Random(0)
    scs = [Scenario(num_drivers=n, driver_capacity=3, speed_kmh=20.0,
                    regime="peak", seed=i)
           for i, n in enumerate([300, 320, 700, 750, 1200, 1300, 400, 900, 1100])]
    objs = [_Obj(f) for f in ("completion", "completion", "completion",
                              "pooling", "pooling", "pooling",
                              "raw", "raw", "raw")]
    objs = pair_by_fleet_band(scs, objs, DEFAULT_FLEET_BANDS)
    sts = pair_by_strength_band(scs, objs, rng, DEFAULT_FLEET_BANDS)

    _check("one strength per cell", len(sts) == len(scs), f"({len(sts)})")
    _check("strengths are non-negative and uncapped-compatible",
           all(s >= 0.0 for s in sts) and max(sts) > 1.0,
           f"(max {max(sts):.2f})")

    per_family: Dict[str, List[str]] = {}
    for ob, s in zip(objs, sts):
        per_family.setdefault(ob.family, []).append(strength_label(s))
    for fam, labs in sorted(per_family.items()):
        _check(f"family {fam!r} is graded with the budget OFF and ON",
               "off" in labs and any(l != "off" for l in labs), str(labs))

    # The two axes have three levels each: a naive round-robin would marry one
    # strength band to one fleet band permanently. Count the 3x3 table over many
    # rounds and require every cell to be populated within a factor of ~2.
    table = {(b, l): 0 for b in range(3) for l in ("off", "mild", "strong")}
    for r in range(400):
        rr = random.Random(r)
        s2 = pair_by_strength_band(scs, objs, rr, DEFAULT_FLEET_BANDS)
        for sc, s in zip(scs, s2):
            table[(band_index(sc, DEFAULT_FLEET_BANDS), strength_label(s))] += 1
    lo, hi = min(table.values()), max(table.values())
    _check("fleet band and strength band stay uncorrelated",
           lo > 0 and hi <= 2.5 * lo, f"(min {lo}, max {hi} over 400 rounds)")

    # Same rng state -> same strengths: a round's grid has to be reproducible.
    a = pair_by_strength_band(scs, objs, random.Random(7), DEFAULT_FLEET_BANDS)
    b = pair_by_strength_band(scs, objs, random.Random(7), DEFAULT_FLEET_BANDS)
    _check("the grid is reproducible from the round's rng", a == b)


def check_group_scoring() -> None:
    """Per-cell standardisation, the two anchors, and the fallback penalty."""
    rows = [[100.0, 5.0, 900.0, 12.0],
            [110.0, 4.0, 800.0, 13.0],
            [90.0, 6.0, 950.0, 11.0]]
    refs = [(105.0, 95.0), (5.5, 4.5), (880.0, 870.0), (12.5, 11.5)]
    evs = group_evals(rows, refs, _OBJECTIVES, _STRENGTHS)

    _check("per-strength bands are the extra axis",
           set(evs[0].per_strength) == {"off", "mild", "strong"},
           str(sorted(evs[0].per_strength)))
    _check("per-family is still the Phase-2 axis",
           set(evs[0].per_family) == {"raw", "completion", "pooling"})

    scaled = [[r[0] * 1000.0] + r[1:] for r in rows]
    srefs = [(refs[0][0] * 1000.0, refs[0][1] * 1000.0)] + list(refs[1:])
    evs2 = group_evals(scaled, srefs, _OBJECTIVES, _STRENGTHS)
    _check("scale-invariance (x1000 on one cell changes nothing)",
           all(abs(a.per_cell[0] - b.per_cell[0]) < 1e-9 for a, b in zip(evs, evs2)))

    # The whole point of the anchors: a round where every live program is worse
    # than not repositioning must not hand out advantages near zero.
    bad = [[1.0, 1.0, 1.0, 1.0], [1.1, 1.1, 1.1, 1.1], [0.9, 0.9, 0.9, 0.9]]
    b_refs = [(50.0, 40.0)] * 4
    evs3 = group_evals(bad, b_refs, _OBJECTIVES, _STRENGTHS)
    _check("beating only the live programs is still NEGATIVE against the anchors",
           all(e.raw_fitness < 0.0 for e in evs3),
           f"(best {max(e.raw_fitness for e in evs3):+.2f})")
    _check("the absolute bar is reported per family",
           all(v == 2.0 for v in evs3[0].family_anchors_above.values()),
           str(evs3[0].family_anchors_above))

    # The fallback penalty subtracts on the advantage scale, and a defer does not.
    # LEGACY PATH: the penalty is off by default now (a crash parks the car, so it
    # scores like the do-nothing baseline on its own), but the term must still work
    # for anyone reproducing a pre-delta-fitness run.
    sc_break = GuardedScorer(compile_repositioner(_code(1.0), name="breaks"))
    sc_break.n_calls, sc_break.n_fallbacks = 100, 50
    sc_defer = GuardedScorer(compile_repositioner(_code(2.0), name="defers"))
    sc_defer.n_calls, sc_defer.n_defers = 100, 50
    evs4 = group_evals(rows[:2], refs, _OBJECTIVES, _STRENGTHS,
                       scorers=[sc_break, sc_defer], fallback_penalty=2.0)
    _check("a broken scorer is charged, a deferring one is not (legacy penalty)",
           abs(evs4[0].fitness - (evs4[0].raw_fitness - 1.0)) < 1e-9
           and abs(evs4[1].fitness - evs4[1].raw_fitness) < 1e-9,
           f"(fallback {evs4[0].fallback_rate:.2f}, defer {evs4[1].defer_rate:.2f})")
    evs4b = group_evals(rows[:2], refs, _OBJECTIVES, _STRENGTHS,
                        scorers=[sc_break, sc_defer])
    _check("by default no penalty is applied at all",
           all(abs(e.fitness - e.raw_fitness) < 1e-12 for e in evs4b),
           f"(fitness {evs4b[0].fitness:.4f} vs raw {evs4b[0].raw_fitness:.4f})")

    # -- the delta fitness's defining properties ------------------------------ #
    # The LAST anchor is the baseline. A scorer whose reward IS the off anchor on
    # every cell added nothing and must score exactly 0 -- not "average of a weak
    # round", which is what the old centred key would have given it.
    off_row = [r[-1] for r in refs]
    tied = group_evals([list(off_row), [x + 5.0 for x in off_row],
                        [x + 9.0 for x in off_row]],
                       refs, _OBJECTIVES, _STRENGTHS)
    _check("matching the do-nothing baseline scores exactly 0",
           abs(tied[0].raw_fitness) < 1e-12,
           f"({tied[0].raw_fitness:+.4f}; peers {tied[1].raw_fitness:+.2f} "
           f"{tied[2].raw_fitness:+.2f})")
    _check("the sign is absolute: helping is +, hurting is -",
           tied[1].raw_fitness > 0.0 and tied[2].raw_fitness > 0.0
           and group_evals([[x - 5.0 for x in off_row], list(off_row)],
                           refs, _OBJECTIVES, _STRENGTHS)[0].raw_fitness < 0.0)

    # A weak round no longer pays. Under the old centred key the best of three
    # programs that all LOSE to doing nothing scored positive; now it cannot.
    all_worse = [[x - 1.0 for x in off_row], [x - 2.0 for x in off_row],
                 [x - 3.0 for x in off_row]]
    _check("the least bad of an all-losing round still scores negative",
           all(e.raw_fitness < 0.0
               for e in group_evals(all_worse, refs, _OBJECTIVES, _STRENGTHS)))

    # Unanimity must not collapse to 0: "everyone beat doing nothing by the same
    # margin" is the degenerate column the |mean|/clip denominator floor exists
    # for. Checked on the primitive, because in a real cell the heuristic anchor
    # is also a group member and breaks the tie by construction.
    from pref_dispatch.llm.group_fitness import delta_advantage  # noqa: PLC0415
    _check("a unanimous column saturates at the clip, it does not read as 0",
           all(abs(delta_advantage(v, [7.0] * 4) - 3.0) < 1e-9 for v in [7.0] * 4)
           and all(abs(delta_advantage(v, [-7.0] * 4) + 3.0) < 1e-9
                   for v in [-7.0] * 4))
    _check("a column where nobody moved the needle is still 0",
           delta_advantage(0.0, [0.0] * 4) == 0.0)
    _check("a crashed rollout floors at -clip inside the deltas",
           delta_advantage(float("nan"), [1.0, 2.0, 3.0]) == -3.0)


def check_crash_parks_the_car() -> None:
    """A raising scorer must leave cars where they are, never borrow the heuristic.

    This is what makes the fallback penalty unnecessary: under the delta fitness
    the do-nothing baseline is worth exactly 0, so a program that crashes on every
    driver prices its own failure. Before this change ``{}`` was returned and
    :mod:`pref_dispatch.reposition` read it as a DEFER, handing the car to the
    built-in demand-gravity kernel -- a crash silently inherited a working policy.
    """
    always_raises = compile_repositioner(
        "def reposition_scores(driver_obs, phi_ep, phi_step, kappa, w):\n"
        "    raise RuntimeError('boom')\n", name="always_raises")
    g = GuardedScorer(always_raises)
    obs = {"self": {"current_region": 3}}

    class _Phi:
        n_regions = 8
    out = g(obs, _Phi(), _Phi(), None, None)
    _check("a raise returns the driver's OWN region, not an empty dict",
           out == {3: 1.0}, str(out))
    _check("and it is still counted as a break, not a defer",
           g.n_fallbacks == 1 and g.n_defers == 0,
           f"(broke {g.n_fallbacks}, deferred {g.n_defers})")

    honest = compile_repositioner(
        "def reposition_scores(driver_obs, phi_ep, phi_step, kappa, w):\n"
        "    return {}\n", name="honest_defer")
    g2 = GuardedScorer(honest)
    out2 = g2(obs, _Phi(), _Phi(), None, None)
    _check("an honest empty return still defers to the heuristic",
           out2 == {} and g2.n_defers == 1 and g2.n_fallbacks == 0,
           f"({out2}, broke {g2.n_fallbacks}, deferred {g2.n_defers})")

    # End to end: scoring only your current region hits reposition.py's stay rule.
    from pref_dispatch.reposition import choose_relocation_targets  # noqa: PLC0415
    import inspect  # noqa: PLC0415
    src = inspect.getsource(choose_relocation_targets)
    _check("reposition.py still stays put when best_r == current_region",
           "if best_r == current_region:" in src and "continue" in src)
    _check("an empty dict still falls through to the demand-gravity heuristic",
           "if got:" in src)


def check_selection_key_and_elites() -> None:
    """``beta = 0`` (2026-08-13): selection IS the pure mean advantage, so
    equal-mean scorers TIE on the key regardless of how lopsided their fairness
    bands are; the reserved per-band elite slot -- not the key -- is what keeps a
    strong-band specialist alive."""
    even = _mk_eval(0.50, {"raw": 0.5, "completion": 0.5},
                    {"off": 0.55, "strong": 0.45})
    off_only = _mk_eval(0.50, {"raw": 0.5, "completion": 0.5},
                        {"off": 1.40, "strong": -0.40})
    _check("beta=0: equal-mean scorers TIE on the key",
           abs(E.selection_score(even) - E.selection_score(off_only)) < 1e-9,
           f"({E.selection_score(even):+.3f} vs {E.selection_score(off_only):+.3f})")
    _check("beta=0: the key equals the fitness exactly",
           abs(E.selection_score(even) - 0.50) < 1e-9)

    # A strong-band specialist that the plain top-mu cut would drop must survive.
    pool = [
        _fake_candidate("allrounder", 0.9, {"raw": 0.9, "completion": 0.9},
                        {"off": 1.2, "strong": 0.6}),
        _fake_candidate("second", 0.8, {"raw": 0.8, "completion": 0.8},
                        {"off": 1.0, "strong": 0.5}),
        _fake_candidate("fairness_specialist", -0.4, {"raw": -0.9, "completion": 0.1},
                        {"off": -1.4, "strong": 1.9}),
        _fake_candidate("completion_specialist", -0.3, {"raw": -1.0, "completion": 1.6},
                        {"off": 0.2, "strong": -0.8}),
    ]
    lines: List[str] = []
    keep = E.select_survivors(pool, _OBJECTIVES, [0.0, 2.0], mu=2,
                              log=lines.append)
    names = [c.name for c in keep]
    _check("the strong-band specialist is reserved an elite slot",
           "fairness_specialist" in names, str(names))
    _check("the weak-family specialist is reserved one too (Phase-2 rule intact)",
           "completion_specialist" in names, str(names))
    _check("the elite reservation is logged with its axis",
           any("survives on strong" in ln for ln in lines),
           str([ln.strip() for ln in lines]))


def _run(client, stub, *, generations=1, mu=2, lam=2, crossover_rate=0.0,
         fresh_per_round=1, batch_fn=None, log=None, checkpoint_fn=None):
    """Drive the real loop with the rollout + anchors stubbed out."""
    orig_roll, orig_anch = E._roll_cell_rewards, E.anchor_reference_rewards
    E._roll_cell_rewards, E.anchor_reference_rewards = stub, _anchor_stub
    try:
        return E.evolve_repositioner_group(
            client, "PROFILE", {}, combiner=None,
            scenarios=_SCENARIOS, objectives=_OBJECTIVES, strengths=_STRENGTHS,
            batch_fn=batch_fn,
            generations=generations, mu=mu, lam=lam,
            crossover_rate=crossover_rate, fresh_per_round=fresh_per_round,
            rng=random.Random(0), log=log or (lambda _s: None),
            checkpoint_fn=checkpoint_fn,
        )
    finally:
        E._roll_cell_rewards, E.anchor_reference_rewards = orig_roll, orig_anch


def check_end_to_end() -> None:
    """Full ``(mu+lambda)`` run: operators, rotating cells, per-round checkpoint.

    Two runs: ``crossover_rate=1.0`` forces every non-fresh slot through the
    crossover branch, ``0.0`` forces them all through mutation. One run cannot
    exercise both deterministically, and a probabilistic check here would be a
    flaky test rather than a stronger one.
    """
    prompts: List[str] = []
    rounds: List[int] = []
    asked: List[int] = []

    def batch_fn(i: int):
        asked.append(i)
        # Rotate the strengths per round so the loop cannot be caching a grid.
        sts = [(s + 0.5 * i) if s else 0.0 for s in _STRENGTHS]
        return list(_SCENARIOS), list(_OBJECTIVES), sts

    for rate in (1.0, 0.0):
        client = ScriptedClient()
        champ = _run(client, RolloutStub(), generations=2, mu=2, lam=3,
                     crossover_rate=rate, fresh_per_round=1, batch_fn=batch_fn,
                     checkpoint_fn=(lambda g, c: rounds.append(g)) if rate == 1.0
                     else None)
        prompts.extend(client.prompts)
        _check(f"run at crossover_rate={rate} returns a champion",
               champ is not None and champ.evaluation is not None,
               f"({champ.name}, selection "
               f"{E.selection_score(champ.evaluation):+.3f})")

    _check("checkpoint fired once per round", rounds == [0, 1, 2], str(rounds))
    _check("cells were re-drawn every round", asked[:3] == [0, 1, 2], str(asked[:3]))
    _check("crossover operator reached",
           any("RECOMBINING the two parent programs" in p for p in prompts))
    _check("mutation operator reached",
           any("Improve your reposition scorer" in p for p in prompts))
    _check("parentless operator reached",
           any("Write ONE reposition scorer that MAXIMISES" in p for p in prompts))
    _check("parents are shown their weakest FAMILY and weakest BAND",
           any("YOUR WEAKEST FAIRNESS BAND" in p and "YOUR WEAKEST FAMILY" in p
               for p in prompts))


def check_clone_kill_and_runtime_repair() -> None:
    """A behavioural clone is eliminated; a breaker gets one repair, then dies."""
    stub = RolloutStub()
    stub.clone = True
    lines: List[str] = []
    _run(ScriptedClient(), stub, generations=1, mu=2, lam=2, log=lines.append)
    _check("behavioural clones are eliminated",
           any("[clone]" in ln and "ELIMINATED" in ln for ln in lines))

    # v1 breaks, its repair (a fresh name) does not -> exactly one repair, survives.
    client2 = ScriptedClient()
    lines2: List[str] = []
    _run(client2, RolloutStub(broken=("v1",)), generations=0, mu=2, lam=1,
         log=lines2.append)
    _check("a breaking scorer triggers exactly one runtime repair",
           sum(1 for ln in lines2 if "one repair attempt" in ln) == 1,
           str([ln for ln in lines2 if "[runtime]" in ln]))
    _check("the repair prompt carries the real fallback cause",
           any("RUNTIME FAILURE" in p for p in client2.prompts))
    _check("a repaired scorer is NOT eliminated",
           not any("still breaks" in ln for ln in lines2))

    # Everything breaks, forever -> the repair does not take, so it is eliminated.
    lines3: List[str] = []
    _run(ScriptedClient(), RolloutStub(broken=("*",)), generations=0, mu=2, lam=1,
         log=lines3.append)
    _check("a scorer that still breaks after its repair is ELIMINATED",
           any("still breaks" in ln for ln in lines3))


def check_yardstick() -> None:
    """The fixed-batch card counts wins against BOTH anchors, not just one."""
    orig_roll, orig_anch = E._roll_cell_rewards, E.anchor_reference_rewards
    stub = RolloutStub()
    E._roll_cell_rewards, E.anchor_reference_rewards = stub, _anchor_stub
    try:
        champ = E._seed_repositioner_candidate(
            _code(1.0), {"skill_name": "champ", "objective": "seeded"})
        card = E.reposition_yardstick(champ, {}, None, _SCENARIOS, _OBJECTIVES,
                                      _STRENGTHS, log=lambda _s: None)
    finally:
        E._roll_cell_rewards, E.anchor_reference_rewards = orig_roll, orig_anch

    # The stub's champion multiplier is 1 + 0.1*len("champ") = 1.5, i.e. above both
    # anchors (1.05 and 0.95) on every cell.
    _check("the card counts both anchors over all cells",
           card["n_cells"] == len(_SCENARIOS)
           and card["beats_heuristic"] == len(_SCENARIOS)
           and card["beats_off"] == len(_SCENARIOS),
           f"(heuristic {card['beats_heuristic']}/{card['n_cells']}, "
           f"off {card['beats_off']}/{card['n_cells']})")
    _check("the card carries both grouping axes",
           set(card["per_strength"]) == {"off", "mild", "strong"}
           and set(card["per_family"]) == {"raw", "completion", "pooling"})
    _check("mean rewards are reported for the champion and both anchors",
           card["champion_rewards"] > card["anchor_rewards"]["heuristic"]
           > card["anchor_rewards"]["off"],
           f"({card['champion_rewards']:.1f} vs "
           f"{card['anchor_rewards']['heuristic']:.1f} vs "
           f"{card['anchor_rewards']['off']:.1f})")


def check_runner_wiring() -> None:
    """The Phase-3 runner builds a THREE-valued batch and checkpoints to phase3."""
    import inspect

    from pref_dispatch.llm import run_phase3_full as R

    src = inspect.getsource(R)
    _check("the runner pairs on BOTH axes",
           "pair_by_fleet_band(" in src and "pair_by_strength_band(" in src)
    _check("the runner's batch_fn returns (scenes, objectives, strengths)",
           "return scs, objs, sts" in src)
    _check("checkpoints go to the phase-3 directory",
           "LeaderCheckpoint(3" in src)
    _check("the frozen combiner's source is passed for the parallel path",
           "combiner_code=combiner_code" in src)


def main() -> None:
    check_strength_pairing()
    check_group_scoring()
    check_crash_parks_the_car()
    check_selection_key_and_elites()
    check_end_to_end()
    check_clone_kill_and_runtime_repair()
    check_yardstick()
    check_runner_wiring()
    print("\n[B3] ALL Phase-3 group-loop offline checks passed (no API key used).")


if __name__ == "__main__":
    main()
