"""Offline, key-free end-to-end check of the Phase-1 group-relative loop (B1).

Runs :func:`pref_dispatch.llm.evolve_skill_group.evolve_skill_group` against a
scripted client and a stubbed rollout, so nothing touches the network or the real
environment. Asserts the properties the loop is supposed to have:

* the objective and ``fitness_code`` chosen in generation 0 are FIXED for the run,
  even when later replies try to change them;
* fitness is standardised WITHIN each scenario (raw scale cannot matter);
* behavioural clones are eliminated;
* a variant that raises at runtime gets one repair and is eliminated if it still
  raises;
* crossover, mutation and parentless-fresh operators all get reached;
* the per-round checkpoint fires once per round;
* the multi-process path reproduces the in-process rows exactly (this last check
  uses the REAL rollout on tiny scenarios -- stubbing it would prove nothing).

Run: ``python -m pref_dispatch.llm._verify_phase1_group``
"""

from __future__ import annotations

import json
import math
import random
from typing import Dict, List, Optional

from pref_dispatch.llm import evolve_skill_group as G
from pref_dispatch.scenario import Scenario

# --------------------------------------------------------------------------- #
# Scripted client + stubbed rollout.                                          #
# --------------------------------------------------------------------------- #

_FOUNDER_CODE = """
def score(driver_obs, order, phi_ep, phi_step):
    return float(order.get('fare', 0.0))


def noop_score(driver_obs, phi_ep, phi_step):
    return 0.0
"""

# Distinct programs so the rollout stub can give each variant a different row.
_VARIANT_CODES = [
    """
def score(driver_obs, order, phi_ep, phi_step):
    return float(order.get('fare', 0.0)) * 2.0


def noop_score(driver_obs, phi_ep, phi_step):
    return 0.1
""",
    """
def score(driver_obs, order, phi_ep, phi_step):
    return float(order.get('fare', 0.0)) - 1.0


def noop_score(driver_obs, phi_ep, phi_step):
    return 0.2
""",
    """
def score(driver_obs, order, phi_ep, phi_step):
    return -float(order.get('fare', 0.0))


def noop_score(driver_obs, phi_ep, phi_step):
    return 0.3
""",
    """
def score(driver_obs, order, phi_ep, phi_step):
    return float(order.get('fare', 0.0)) * 0.5 + 1.0


def noop_score(driver_obs, phi_ep, phi_step):
    return 0.4
""",
]

_FOUNDER_FITNESS = "def fitness(metrics):\n    return float(metrics.get('revenue', 0.0))\n"
_HIJACK_FITNESS = "def fitness(metrics):\n    return -1.0\n"

# Two programs for the REAL-rollout parallel check. Unlike the stub codes above
# these have to actually dispatch: a skill whose noop outscores every order
# assigns nothing, and the resulting all-zero metrics would make the
# parallel-equals-sequential comparison pass without measuring anything.
_REAL_CODES = [
    ("def score(driver_obs, order, phi_ep, phi_step):\n"
     "    if not _feasible(driver_obs, order):\n"
     "        return -1e9\n"
     "    scale = float(phi_step.mean_solo_time) or float(phi_ep.scale) or 1.0\n"
     "    pickup = _pickup_time(driver_obs, order, phi_ep.dist) / scale\n"
     "    ride = _solo_time(order, phi_ep.dist) / scale\n"
     "    return ride - 0.5 * pickup\n\n\n"
     "def noop_score(driver_obs, phi_ep, phi_step):\n"
     "    return 0.2\n"),
    # Same feasibility rule, opposite preference: short rides, pickup-averse.
    ("def score(driver_obs, order, phi_ep, phi_step):\n"
     "    if not _feasible(driver_obs, order):\n"
     "        return -1e9\n"
     "    scale = float(phi_step.mean_solo_time) or float(phi_ep.scale) or 1.0\n"
     "    pickup = _pickup_time(driver_obs, order, phi_ep.dist) / scale\n"
     "    ride = _solo_time(order, phi_ep.dist) / scale\n"
     "    return -ride - 2.0 * pickup\n\n\n"
     "def noop_score(driver_obs, phi_ep, phi_step):\n"
     "    return -5.0\n"),
]


class ScriptedClient:
    """Cycles through the variant programs; records every prompt it was sent."""

    def __init__(self) -> None:
        self.prompts: List[str] = []
        self.n = 0

    def complete(self, system: str, user: str, temperature=None) -> str:
        self.prompts.append(user)
        self.n += 1
        first = "# FIXED OBJECTIVE" not in user
        code = _FOUNDER_CODE if first else _VARIANT_CODES[self.n % len(_VARIANT_CODES)]
        obj = {
            "skill_name": f"v{self.n}",
            # Every reply after the founder tries to HIJACK the objective and the
            # fitness. The loop must ignore both.
            "objective": "founder objective" if first else "HIJACKED objective",
            "mechanism": f"mechanism variant {self.n}",
            "differs_from": "differs from the others by its ranking term",
            "description": "Scores orders by fare, waiting when nothing clears the bar.",
            "fitness_code": _FOUNDER_FITNESS if first else _HIJACK_FITNESS,
            "fitness_rationale": "Revenue is the objective, so revenue is the fitness.",
            "code": code,
        }
        return json.dumps(obj)


_SCENARIOS = [
    Scenario(num_drivers=300, driver_capacity=3, speed_kmh=20.0, regime="peak", seed=1),
    Scenario(num_drivers=300, driver_capacity=3, speed_kmh=20.0, regime="offpeak", seed=2),
    Scenario(num_drivers=1200, driver_capacity=3, speed_kmh=20.0, regime="peak", seed=3),
    Scenario(num_drivers=1200, driver_capacity=3, speed_kmh=20.0, regime="offpeak", seed=4),
]


class RolloutStub:
    """Deterministic fake rollout: revenue depends on the variant AND the scene.

    Scene levels differ by 1000x on purpose -- if the loop were averaging raw
    fitness instead of standardising per scenario, the big scene would decide
    everything and the assertions below would fail.
    """

    LEVELS = {1: 10.0, 2: 10_000.0, 3: 3.0, 4: 30_000.0}

    def __init__(self) -> None:
        self.raisers: Dict[str, int] = {}   # code fragment -> times still to raise
        self.calls = 0

    def __call__(self, skill, scenario) -> Dict[str, float]:
        self.calls += 1
        code = getattr(skill, "code", "") or ""
        for frag, left in list(self.raisers.items()):
            if frag in code and left > 0:
                self.raisers[frag] = left - 1
                raise ZeroDivisionError("float division by zero")
        # A per-skill multiplier derived from its noop offset, so different
        # programs get genuinely different rows.
        try:
            mult = 1.0 + float(skill.noop_score({}, None, None))
        except Exception:  # noqa: BLE001
            mult = 1.0
        return {"revenue": RolloutStub.LEVELS[scenario.seed] * mult}


def _check(label: str, ok: bool, detail: str = "") -> None:
    if not ok:
        raise AssertionError(f"{label} FAILED {detail}")
    print(f"[B1] {label} OK {detail}".rstrip())


# --------------------------------------------------------------------------- #
# Checks.                                                                     #
# --------------------------------------------------------------------------- #

def check_group_scoring() -> None:
    """Within-scenario standardisation: scale-free, discriminating, dead-round-aware."""
    rows = [[100.0, 5.0, 900.0, 12.0],
            [110.0, 4.0, 800.0, 13.0],
            [90.0, 6.0, 950.0, float("nan")]]
    mets = [[{} for _ in _SCENARIOS] for _ in rows]
    evs = G.group_evals(rows, mets, _SCENARIOS, errors=[None, None, "ZeroDivisionError"])

    col0 = sum(e.per_scenario_adv[0] for e in evs)
    _check("column advantages sum to zero", abs(col0) < 1e-9, f"({col0:.2e})")

    scaled = [[r[0] * 1000.0] + r[1:] for r in rows]
    evs2 = G.group_evals(scaled, mets, _SCENARIOS)
    same = all(abs(a.per_scenario_adv[0] - b.per_scenario_adv[0]) < 1e-9
               for a, b in zip(evs, evs2))
    _check("scale-invariance (x1000 on one scene changes nothing)", same)

    _check("raising variant is penalised", evs[2].fitness < min(e.fitness for e in evs[:2]),
           f"({evs[2].fitness:+.2f} vs {evs[0].fitness:+.2f}/{evs[1].fitness:+.2f})")
    _check("invalid_rate counted", abs(evs[2].invalid_rate - 0.25) < 1e-9)

    # Every variant raises on one scene -> the column is all-nan, advantages are all
    # 0.0, and ONLY the explicit penalty distinguishes it from a healthy tie.
    dead_col = [[float("nan"), 5.0, 900.0, 12.0],
                [float("nan"), 4.0, 800.0, 13.0],
                [float("nan"), 6.0, 950.0, 14.0]]
    evs3 = G.group_evals(dead_col, mets, _SCENARIOS)
    _check("all-raising column: only the explicit penalty registers",
           all(abs(e.per_scenario_adv[0]) < 1e-12 for e in evs3)
           and all(e.fitness < e.raw_fitness for e in evs3))

    flat = [[7.0] * 4 for _ in range(3)]
    evs4 = G.group_evals(flat, mets, _SCENARIOS)
    _check("dead round scores 0 and reports tie_rate 1.0",
           all(abs(e.raw_fitness) < 1e-12 for e in evs4)
           and all(t == 1.0 for t in evs4[0].band_tie_rate.values()))

    bands = set(evs[0].per_band)
    _check("bands are the diversity axis", bands == {"fleet200-500", "fleet1000-1500"},
           str(sorted(bands)))


def check_selection_key() -> None:
    """``beta = 0`` (2026-08-13): selection IS the pure mean advantage, so
    equal-mean variants TIE on the key regardless of how lopsided their bands
    are. The reserved per-band elite slot -- not the key -- is what keeps a
    specialist that is the best at one scale alive."""
    strong_even = G.SkillGroupEval(fitness=0.50, raw_fitness=0.50,
                                   per_band={"a": 0.55, "b": 0.45})
    strong_lopsided = G.SkillGroupEval(fitness=0.50, raw_fitness=0.50,
                                       per_band={"a": 1.40, "b": -0.40})
    _check("beta=0: equal-mean variants TIE on the key",
           abs(G.selection_score(strong_even) - G.selection_score(strong_lopsided)) < 1e-9,
           f"({G.selection_score(strong_even):+.3f} vs "
           f"{G.selection_score(strong_lopsided):+.3f})")
    _check("beta=0: the key equals the fitness exactly",
           abs(G.selection_score(strong_even) - 0.50) < 1e-9)


def check_end_to_end() -> None:
    """Full (mu+lambda) run against the scripted client and stubbed rollout.

    Two runs: ``crossover_rate=1.0`` forces every non-fresh slot through the
    crossover branch, ``0.0`` forces them all through mutation. One run cannot
    exercise both deterministically, and a probabilistic check here would be a
    flaky test rather than a stronger one.
    """
    orig = G.rollout_skill_on_scenario
    prompts: List[str] = []
    champs = []
    rounds: List[int] = []

    for rate in (1.0, 0.0):
        client = ScriptedClient()
        G.rollout_skill_on_scenario = RolloutStub()
        seen: List[int] = []
        try:
            champs.append(G.evolve_skill_group(
                client, "PROFILE",
                objective_hint="maximise revenue",
                scenarios=_SCENARIOS,
                generations=2, mu=2, lam=3,
                crossover_rate=rate,
                fresh_per_round=1,        # exercises the parentless branch too
                rng=random.Random(0),
                checkpoint_fn=lambda i, c: seen.append(i),
                log=lambda s: None,
            ))
        finally:
            G.rollout_skill_on_scenario = orig
        prompts.extend(client.prompts)
        if rate == 1.0:
            rounds = seen

    champ = champs[0]
    _check("run returns a champion", champ is not None and champ.name.startswith("v"))
    _check("objective stayed FIXED from generation 0",
           all(c.meta["objective"] == "founder objective" for c in champs),
           f"({champ.meta['objective']!r})")
    _check("fitness_code stayed FIXED from generation 0",
           all(c.meta["fitness_code"] == _FOUNDER_FITNESS for c in champs))
    _check("champion recorded its mechanism", bool(champ.meta.get("mechanism")))
    _check("checkpoint fired once per round", rounds == [0, 1, 2], str(rounds))

    _check("crossover operator reached",
           any("COMBINE the two dispatch skills" in p for p in prompts))
    _check("mutation operator reached",
           any("Improve ONE existing dispatch skill" in p for p in prompts))
    _check("parentless operator reached",
           any("Write a NEW implementation of the fixed objective" in p for p in prompts))
    _check("later prompts carry the FIXED fitness",
           all(_FOUNDER_FITNESS.strip() in p
               for p in prompts if "# FIXED OBJECTIVE" in p))


def check_clone_kill_and_runtime_repair() -> None:
    """A behavioural clone is eliminated; a raiser gets one repair, then dies."""
    stub = RolloutStub()
    orig = G.rollout_skill_on_scenario
    G.rollout_skill_on_scenario = stub

    class CloneClient(ScriptedClient):
        """Always returns the SAME program, so every variant is a behavioural clone."""

        def complete(self, system: str, user: str, temperature=None) -> str:
            obj = json.loads(super().complete(system, user, temperature))
            obj["code"] = _FOUNDER_CODE
            return json.dumps(obj)

    lines: List[str] = []
    try:
        G.evolve_skill_group(
            CloneClient(), "PROFILE", scenarios=_SCENARIOS,
            generations=1, mu=2, lam=2, crossover_rate=0.0, fresh_per_round=1,
            rng=random.Random(0), log=lines.append,
        )
    finally:
        G.rollout_skill_on_scenario = orig
    _check("behavioural clones are eliminated",
           any("[clone]" in ln and "ELIMINATED" in ln for ln in lines))

    # Now: the founder program raises on the first four rollouts (one whole round's
    # batch), the repair's program does not -> one repair, then it survives.
    stub2 = RolloutStub()
    stub2.raisers = {"return float(order.get('fare', 0.0))\n": 4}
    G.rollout_skill_on_scenario = stub2
    lines2: List[str] = []
    client2 = ScriptedClient()
    try:
        G.evolve_skill_group(
            client2, "PROFILE", scenarios=_SCENARIOS,
            generations=0, mu=2, lam=1, crossover_rate=0.0, fresh_per_round=1,
            rng=random.Random(0), log=lines2.append,
        )
    finally:
        G.rollout_skill_on_scenario = orig
    _check("a raising variant triggers exactly one runtime repair",
           sum(1 for ln in lines2 if "one repair attempt" in ln) == 1,
           str([ln for ln in lines2 if "runtime" in ln]))
    _check("the repair prompt carries the real exception and the no-fallback rule",
           any("ZeroDivisionError" in p and "a skill has NO fallback" in p
               for p in client2.prompts))

    # And a variant that keeps raising is eliminated for the round.
    stub3 = RolloutStub()
    stub3.raisers = {"def score(": 10_000}   # everything raises, forever
    G.rollout_skill_on_scenario = stub3
    lines3: List[str] = []
    try:
        G.evolve_skill_group(
            ScriptedClient(), "PROFILE", scenarios=_SCENARIOS,
            generations=0, mu=2, lam=1, crossover_rate=0.0, fresh_per_round=1,
            rng=random.Random(0), log=lines3.append,
        )
    finally:
        G.rollout_skill_on_scenario = orig
    _check("a variant that still raises after repair is ELIMINATED",
           any("still raises" in ln for ln in lines3))


def check_parallel_matches_sequential() -> None:
    """Spreading the round's rollouts over processes must not move a number.

    Unlike the checks above this one uses the REAL rollout on tiny capped-order
    scenarios: the point is precisely that a worker reproduces what the in-process
    loop produces, so stubbing the thing under test would prove nothing. Two
    variants x two scenes at 120 drivers / 150 orders is a few seconds.
    """
    from pref_dispatch.llm.parallel import (
        NotParallelizable,
        parallel_skill_group_rows,
        skill_group_payload,
    )
    from pref_dispatch.llm.sandbox import compile_skill
    from pref_dispatch.llm.fitness_eval import rollout_skill_on_scenario

    def _cand(name: str, code: str) -> G.Candidate:
        return G.Candidate(
            meta={"skill_name": name, "code": code},
            skill=compile_skill(code, name=name),
            fitness_fn=lambda m: (100.0 * float(m["service_rate"])
                                  + 0.01 * float(m["revenue"])),
        )

    pool = [_cand("a", _REAL_CODES[0]), _cand("b", _REAL_CODES[1])]
    scs = [
        Scenario(num_drivers=120, driver_capacity=4, speed_kmh=35.0,
                 regime="offpeak", split="train", order_limit=150, seed=s)
        for s in (0, 1)
    ]

    seq = [[float(c.fitness_fn(rollout_skill_on_scenario(c.skill, sc)))
            for sc in scs] for c in pool]
    # A vacuous row (every cell 0, or both variants identical) would make the
    # equality below prove nothing, so pin that the batch actually discriminates.
    _check("the parallel check's own batch is non-degenerate",
           any(v != 0.0 for row in seq for v in row) and seq[0] != seq[1],
           f"({seq})")
    par_raw, par_met, par_err = parallel_skill_group_rows(
        pool, scs, workers=2, log=lambda *_: None)
    _check("parallel raw fitness == sequential", par_raw == seq,
           f"({par_raw} vs {seq})")
    _check("parallel returns full metrics dicts, not reduced scalars",
           all(isinstance(m, dict) and "revenue" in m
               for row in par_met for m in row))
    _check("no spurious errors on a healthy round", par_err == [None, None],
           str(par_err))

    # Positional keys: two variants of one skill routinely share a name, and a
    # collision would silently make both rows measure the same program.
    same = [_cand("dup", _REAL_CODES[0]), _cand("dup", _REAL_CODES[1])]
    keys = [p["name"] for p in skill_group_payload(same)]
    _check("payload keys are positional, so same-named variants stay distinct",
           keys == ["c0", "c1"], str(keys))

    # A variant with no source cannot be described for a worker: the whole call
    # refuses, so the caller drops to the in-process loop with nothing wasted.
    naked = _cand("naked", _REAL_CODES[0])
    naked.skill.code = ""
    try:
        parallel_skill_group_rows([naked], scs, workers=2, log=lambda *_: None)
    except NotParallelizable:
        _check("a source-less variant is refused up front", True)
    else:
        raise AssertionError("a source-less variant must raise NotParallelizable")

    # A variant whose FITNESS raises must land as nan + a named error on its own
    # row only, exactly as the in-process loop scores it -- the fitness runs on
    # this side of the process boundary, so this is the one failure mode the
    # worker cannot report for us.
    bad = _cand("bad_fitness", _REAL_CODES[0])
    bad.fitness_fn = lambda m: 1.0 / 0.0
    raw_b, met_b, err_b = parallel_skill_group_rows(
        [bad, pool[0]], scs, workers=2, log=lambda *_: None)
    _check("a raising FITNESS becomes nan on its own row only",
           all(math.isnan(v) for v in raw_b[0]) and raw_b[1] == seq[0]
           and all(m is None for m in met_b[0])
           and err_b[0] is not None and "ZeroDivisionError" in err_b[0]
           and err_b[1] is None,
           str(err_b[0]))

    # One unbuildable variant must fail its OWN cells only. It is stored as the
    # exception worker-side, so the healthy variant sharing the payload still runs.
    from pref_dispatch.llm import parallel as P

    broken = [{"name": "c0", "kind": "code", "code": "def score(:\n"},
              {"name": "c1", "kind": "code", "code": _REAL_CODES[0]}]
    P._GROUP_CACHE.clear()
    basis = P._worker_group_skills(broken)
    _check("an uncompilable variant poisons only its own cells",
           isinstance(basis["c0"], BaseException)
           and not isinstance(basis["c1"], BaseException))


def main() -> None:
    check_group_scoring()
    check_selection_key()
    check_end_to_end()
    check_clone_kill_and_runtime_repair()
    check_parallel_matches_sequential()
    print("\n[B1] ALL Phase-1 group-loop offline checks passed (no API key used).")


if __name__ == "__main__":
    main()
