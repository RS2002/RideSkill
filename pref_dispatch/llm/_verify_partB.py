"""Offline end-to-end verification for Part B training drivers (NO LLM key).

Drives the three Part-B training loops with a SCRIPTED fake client that returns
hand-written, sandbox-valid JSON for each prompt kind, on a tiny env, so the whole
objective-randomized paradigm runs key-free (per MEMORY ``never-write-api-key-to-repo``).

Checks:
  B0  the objective sampler yields a key-free distribution of ``w`` (raw/weights/
      structural families), each ``w(event)`` finite, spec text always present.
  B1  run_phase1 (1a) evolves ONE skill per researcher direction incl. the FAIRNESS
      direction, keeps them all (directed provenance), and hands a superset basis to
      the QD loop; every directed skill carries an NL objective + fitness rationale.
  B1b the v3 fill->replace loop keeps exploring at the cap: the repository is held
      at exactly ``max_skills``, redundant proposals lose the competition, protected
      (seed / directed) skills are never evicted, both stop conditions fire, and an
      evicted skill's frozen artifacts leave the flat ``load_basis`` scan path.
  B2  evolve_combiner_objectives scores a combiner across a BATCH of sampled
      objectives (reads ``w``), returns a finite fitness, and the winner validates
      against the frozen skill names.
  B2b the objective-blindness DIAGNOSTIC still separates a constant combiner
      (blindness ~1) from a w-flipping one (~0), and -- since v6 deleted the
      penalty -- no longer moves fitness at all.
  B2d what the deleted anti-harm penalty was for, done by the group instead: of two
      combiners that move the fleet by the SAME amount, the one that moves it onto
      the worse skill ranks below the one that moves it onto the better skill, with
      no penalty term involved.
  B2e the GRPO-style GROUP-RELATIVE fitness (the training redesign): 2x objectives
      rank byte-identically (internal scale), good/mid/bad candidates order with a
      real spread (no ruler saturation), single-skill anchors rank as expected, and
      the fallback penalty still subtracts.
  B2f the selection key is the PURE GRPO mean advantage (``beta = 0``,
      2026-08-13): a flat mediocre candidate loses to a strong one (no
      mediocrity hole), equal-mean specialist and all-rounder tie, and a flat
      mediocre candidate does
      NOT slip into a hole above the champion, the crasher stays behind, and the
      accept rule admits a weak-family-fix candidate while rejecting the mediocre
      one (pure arithmetic on :func:`selection_score`, no rollouts; its fixtures
      keep the v1 champion's historical family names, including the since-retired
      ``nonlinear``, because the rule is family-agnostic).
  B2g the two sampling fixes for the 2026-08-09 gate (19/30, completion 1/6): the
      structural sub-families are drawn ROUND-ROBIN so none comes out 0
      (pooling did), and each family's draws are PAIRED across different fleet
      bands so none is trained at one scale only (completion was, at fleet 938).
  B3  evolve_repositioner_objectives scores a scorer across random objectives AND
      random fairness strengths; batch fitness is finite and the winner validates.

All rollouts use a ~120-car capped-order scenario so the script finishes fast.
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional

from pref_dispatch.llm.objective_sampler import (
    ObjectiveSampler,
    sample_strengths,
)
from pref_dispatch.scenario import Scenario


# --------------------------------------------------------------------------- #
# Scripted fake client: valid JSON per prompt kind, no network, no key.        #
# --------------------------------------------------------------------------- #
# Minimal but sandbox-valid function bodies in the FINAL-version signatures
# (Part A): skills score(driver_obs, order, phi_ep, phi_step); combiner
# skill_scores(driver_obs, phi_ep, phi_step, w); repositioner
# reposition_scores(driver_obs, phi_ep, phi_step, kappa, w).

_SKILL_CODE = (
    "def score(driver_obs, order, phi_ep, phi_step):\n"
    "    if not _feasible(driver_obs, order):\n"
    "        return -1e9\n"
    "    scale = float(phi_step.mean_solo_time) or float(phi_ep.scale) or 1.0\n"
    "    pickup = _pickup_time(driver_obs, order, phi_ep.dist) / scale\n"
    "    ride = _solo_time(order, phi_ep.dist) / scale\n"
    "    return ride - 0.5 * pickup\n\n\n"
    "def noop_score(driver_obs, phi_ep, phi_step):\n"
    "    return 0.2\n"
)
# The MIRROR of _SKILL_CODE: it prefers the orders _SKILL_CODE ranks last (short
# rides, tolerating a long pickup). Both genuinely SERVE -- unlike _BAD_SKILL_CODE,
# which idles -- so an equal blend of the two is a real policy that beats either
# alone. test_b2e_delta_sign_is_absolute needs exactly that: a basis where the
# baseline is above both candidates, so both can read negative in one round.
_SLOW_SKILL_CODE = (
    "def score(driver_obs, order, phi_ep, phi_step):\n"
    "    if not _feasible(driver_obs, order):\n"
    "        return -1e9\n"
    "    scale = float(phi_step.mean_solo_time) or float(phi_ep.scale) or 1.0\n"
    "    pickup = _pickup_time(driver_obs, order, phi_ep.dist) / scale\n"
    "    ride = _solo_time(order, phi_ep.dist) / scale\n"
    "    return pickup - 0.5 * ride\n\n\n"
    "def noop_score(driver_obs, phi_ep, phi_step):\n"
    "    return 0.2\n"
)
_FITNESS_CODE = (
    "def fitness(metrics):\n"
    "    return (100.0 * metrics['service_rate']\n"
    "            + 0.01 * metrics['revenue']\n"
    "            - metrics['mean_service_time'])\n"
)
# A fairness-leaning fitness (used for the fairness direction): reward equity via
# a low income Gini and a lifted income floor, grounded in the real KPIs.
_FAIRNESS_FITNESS_CODE = (
    "def fitness(metrics):\n"
    "    return (50.0 * metrics['service_rate']\n"
    "            - 60.0 * metrics['income_gini']\n"
    "            + 0.5 * metrics['income_min'])\n"
)
# The combiner returns the FIXED basis skill names as keys (it cannot reference a
# global; real combiners hard-code the frozen names too). The verify basis is the
# two named skills built in main(): "eff" and "fair".
_COMBINER_CODE = (
    "def skill_scores(driver_obs, phi_ep, phi_step, w):\n"
    "    scores = {'eff': 0.5, 'fair': 0.5}\n"
    "    # Guarded probe of the objective: lean toward 'eff' when w rewards a\n"
    "    # populated assignment event, else keep the balanced blend.\n"
    "    if w is not None:\n"
    "        try:\n"
    "            ev = {'assigned_orders': [1],\n"
    "                  'assigned_party_sizes': {1: 2},\n"
    "                  'assigned_solo_times': {1: 5.0},\n"
    "                  'assigned_service_times': {1: 3.0},\n"
    "                  'completed_orders': [], 'picked_up_orders': [],\n"
    "                  'distance_moved': 0.0, 'time_moved': 0.0,\n"
    "                  'is_empty_move': False, 'is_idle_wait': False,\n"
    "                  'extra_detour_time': 0.0}\n"
    "            gain = float(w(ev))\n"
    "            scores['eff'] = 0.5 + max(0.0, gain)\n"
    "        except Exception:\n"
    "            pass\n"
    "    return scores\n"
)
# A COMBINER THAT NEVER ACTS ON THE OBJECTIVE: constant scores, ignores ``w``.
# The B2b blindness DIAGNOSTIC must still read it as ~1 (the previous frozen
# combiner read ``w`` into a lean that never flipped the argmax -- w-delta = 0 in
# every gate cell), even though v6 no longer charges fitness for it.
_BLIND_COMBINER_CODE = (
    "def skill_scores(driver_obs, phi_ep, phi_step, w):\n"
    "    return {'eff': 0.7, 'fair': 0.3}\n"
)
# A combiner that GENUINELY FLIPS with the objective: probe ``w`` on a
# completion-shaped event vs a seating-shaped event; a completion-gated reward
# pays the former, a pooling/assignment reward pays the latter, so the argmax
# switches between 'fair' and 'eff' across the batch's objectives.
_DIFF_COMBINER_CODE = (
    "def skill_scores(driver_obs, phi_ep, phi_step, w):\n"
    "    if w is None:\n"
    "        return {'eff': 0.5, 'fair': 0.5}\n"
    "    try:\n"
    "        comp_ev = {'assigned_orders': [], 'assigned_party_sizes': {},\n"
    "                   'assigned_solo_times': {}, 'assigned_service_times': {},\n"
    "                   'completed_orders': [1], 'picked_up_orders': [],\n"
    "                   'distance_moved': 0.0, 'time_moved': 0.0,\n"
    "                   'is_empty_move': False, 'is_idle_wait': False,\n"
    "                   'extra_detour_time': 0.0}\n"
    "        pool_ev = {'assigned_orders': [1], 'assigned_party_sizes': {1: 3},\n"
    "                   'assigned_solo_times': {1: 5.0}, 'assigned_service_times': {1: 3.0},\n"
    "                   'completed_orders': [], 'picked_up_orders': [],\n"
    "                   'distance_moved': 0.0, 'time_moved': 0.0,\n"
    "                   'is_empty_move': False, 'is_idle_wait': False,\n"
    "                   'extra_detour_time': 0.0}\n"
    "        v_comp, v_pool = float(w(comp_ev)), float(w(pool_ev))\n"
    "        if v_comp > v_pool:\n"
    "            return {'eff': 0.2, 'fair': 0.9}   # completion-gated objective\n"
    "        return {'eff': 0.9, 'fair': 0.2}       # pooling / assignment objective\n"
    "    except Exception:\n"
    "        return {'eff': 0.5, 'fair': 0.5}\n"
)
_REPOSITION_CODE = (
    "def reposition_scores(driver_obs, phi_ep, phi_step, kappa, w):\n"
    "    self_obs = driver_obs['self']\n"
    "    r = int(self_obs['current_region'])\n"
    "    pts = driver_obs['relocation_points']\n"
    "    out = {}\n"
    "    if 0 <= r < len(pts):\n"
    "        # Chase netted demand in the current region; scale-free base score.\n"
    "        try:\n"
    "            out[r] = float(kappa.eff_demand[r])\n"
    "        except Exception:\n"
    "            out[r] = 0.0\n"
    "    return out\n"
)
# A DELIBERATELY BAD skill: every order scores 0 while the no-op scores 10, so a
# driver following it waits (softmax over {orders}+{NOOP} -> noop ~ 1). A combiner
# that flips the fleet onto it under an objective earns LESS than the rest of the
# group on that objective -- exactly the failure the last gate showed (completion
# steering net-negative at fleet1000_offpeak_full).
_BAD_SKILL_CODE = (
    "def score(driver_obs, order, phi_ep, phi_step):\n"
    "    if not _feasible(driver_obs, order):\n"
    "        return -1e9\n"
    "    return 0.0\n\n\n"
    "def noop_score(driver_obs, phi_ep, phi_step):\n"
    "    return 10.0\n"
)
# Shape-probing combiners over an {'eff','far'} basis (eff = good, far = idles the
# fleet). Both read w the same way; they differ ONLY in where the completion flip
# lands, so the group ranking has to separate them on WHERE they steer.
_WRONG_FLIP_COMBINER_CODE = (
    "def skill_scores(driver_obs, phi_ep, phi_step, w):\n"
    "    if w is None:\n"
    "        return {'eff': 0.5, 'far': 0.5}\n"
    "    try:\n"
    "        comp_ev = {'assigned_orders': [], 'assigned_party_sizes': {},\n"
    "                   'assigned_solo_times': {}, 'assigned_service_times': {},\n"
    "                   'completed_orders': [1], 'picked_up_orders': [],\n"
    "                   'distance_moved': 0.0, 'time_moved': 0.0,\n"
    "                   'is_empty_move': False, 'is_idle_wait': False,\n"
    "                   'extra_detour_time': 0.0}\n"
    "        pool_ev = {'assigned_orders': [1], 'assigned_party_sizes': {1: 3},\n"
    "                   'assigned_solo_times': {1: 5.0}, 'assigned_service_times': {1: 3.0},\n"
    "                   'completed_orders': [], 'picked_up_orders': [],\n"
    "                   'distance_moved': 0.0, 'time_moved': 0.0,\n"
    "                   'is_empty_move': False, 'is_idle_wait': False,\n"
    "                   'extra_detour_time': 0.0}\n"
    "        v_comp, v_pool = float(w(comp_ev)), float(w(pool_ev))\n"
    "        if v_comp > v_pool:\n"
    "            return {'eff': 0.1, 'far': 0.9}   # completion-gated: WRONG flip\n"
    "        return {'eff': 0.9, 'far': 0.1}       # pooling: flip is fine\n"
    "    except Exception:\n"
    "        return {'eff': 0.5, 'far': 0.5}\n"
)
_RIGHT_FLIP_COMBINER_CODE = (
    "def skill_scores(driver_obs, phi_ep, phi_step, w):\n"
    "    if w is None:\n"
    "        return {'eff': 0.5, 'far': 0.5}\n"
    "    try:\n"
    "        comp_ev = {'assigned_orders': [], 'assigned_party_sizes': {},\n"
    "                   'assigned_solo_times': {}, 'assigned_service_times': {},\n"
    "                   'completed_orders': [1], 'picked_up_orders': [],\n"
    "                   'distance_moved': 0.0, 'time_moved': 0.0,\n"
    "                   'is_empty_move': False, 'is_idle_wait': False,\n"
    "                   'extra_detour_time': 0.0}\n"
    "        pool_ev = {'assigned_orders': [1], 'assigned_party_sizes': {1: 3},\n"
    "                   'assigned_solo_times': {1: 5.0}, 'assigned_service_times': {1: 3.0},\n"
    "                   'completed_orders': [], 'picked_up_orders': [],\n"
    "                   'distance_moved': 0.0, 'time_moved': 0.0,\n"
    "                   'is_empty_move': False, 'is_idle_wait': False,\n"
    "                   'extra_detour_time': 0.0}\n"
    "        v_comp, v_pool = float(w(comp_ev)), float(w(pool_ev))\n"
    "        if v_comp > v_pool:\n"
    "            return {'eff': 0.9, 'far': 0.1}   # completion-gated: RIGHT flip\n"
    "        return {'eff': 0.9, 'far': 0.1}\n"
    "    except Exception:\n"
    "        return {'eff': 0.5, 'far': 0.5}\n"
)


# Constant-quality combiners over the {'eff','far'} basis for the B2e GROUP tests:
# ``balanced`` serves HALF the fleet every step (deterministic driver_id parity --
# a genuinely MIDDLING no-w policy whose reward sits between the eff-always and
# far-always anchors), ``far_lean`` idles almost always (bad under every serving
# objective). Neither reads ``w`` -- they exist to give the group a clear quality
# ordering.
_BALANCED_COMBINER_CODE = (
    "def skill_scores(driver_obs, phi_ep, phi_step, w):\n"
    "    if int(driver_obs['self']['driver_id']) % 2 == 0:\n"
    "        return {'eff': 1.0, 'far': 0.0}\n"
    "    return {'eff': 0.0, 'far': 1.0}\n"
)
_FAR_LEAN_COMBINER_CODE = (
    "def skill_scores(driver_obs, phi_ep, phi_step, w):\n"
    "    return {'eff': 0.2, 'far': 0.8}\n"
)
# A combiner that crashes every step: nothing usable -> the adapter falls back to
# the default skill on EVERY call. Used to verify the fallback penalty still
# subtracts from the group-relative fitness.
_CRASHING_COMBINER_CODE = (
    "def skill_scores(driver_obs, phi_ep, phi_step, w):\n"
    "    raise RuntimeError('intentional crash for the B2e fallback test')\n"
)


class ScriptedClient:
    """Returns valid JSON for skill / combiner / repositioner prompts by kind.

    Detects the prompt kind from marker strings the prompt builders emit, so ONE
    client drives all three Part-B loops. Never touches the network or a key.

    ``combiner_codes`` cycles the combiner branch through several DIFFERENT
    programs instead of returning one fixed program forever. The evolution loop's
    behavioural clone-kill (see ``_score_pool``) eliminates every candidate whose
    rollout row repeats an earlier one, so a client that answers every combiner
    prompt identically leaves exactly ONE survivor per round -- which silently
    disables anything the loop only does with >= 2 parents (crossover). Tests that
    exercise those paths pass distinct programs here.

    ``skill_codes`` does the same for the skill branch, which the Phase-1 group
    loop needs for the same reason. It defaults to None -- i.e. one fixed program
    forever -- because the QD fill/replace tests depend on proposals being
    behavioural DUPLICATES, which is exactly what they are asserting gets pruned."""

    def __init__(self, combiner_codes=None, skill_codes=None):
        self.calls = 0
        self._skill_i = 0
        self._combiner_codes = list(combiner_codes) if combiner_codes else None
        self._combiner_i = 0
        self._skill_codes = list(skill_codes) if skill_codes else None

    def complete(self, system: str, user: str, *, temperature=None) -> str:
        self.calls += 1
        # Repositioner: its contract names reposition_scores / relocation regions.
        if "reposition_scores" in user or "reposition scorer" in user.lower():
            return json.dumps({
                "reposition_understanding": "Send idle cars toward netted near-future "
                "demand in the current region, discounted by cruise distance.",
                "skill_name": "demand_chaser",
                "objective": "Cruise idle cars toward net demand.",
                "description": "Scores the current region by netted demand; defers "
                "otherwise. Keeps cars put when nothing is clearly better.",
                "code": _REPOSITION_CODE,
            })
        # Combiner: its contract names skill_scores.
        if "skill_scores" in user:
            if self._combiner_codes:
                i = self._combiner_i % len(self._combiner_codes)
                self._combiner_i += 1
                return json.dumps({
                    "combiner_name": f"scripted_combiner_{i}",
                    "strategy": f"Scripted variant {i} (behaviourally distinct).",
                    "description": "Blends the two skills with variant-specific "
                    "weights so the round's rollout rows differ.",
                    "code": self._combiner_codes[i],
                })
            return json.dumps({
                "combiner_name": "objective_aware_dispatcher",
                "strategy": "Probe w to lean the blend toward the objective.",
                "description": "Idle/loaded cars get a balanced blend; when the "
                "objective w rewards a populated assignment, the first skill is "
                "boosted. Falls back to balanced weights when w is None.",
                "code": _COMBINER_CODE,
            })
        # Otherwise a lower skill (authors fitness + score). Alternate the fitness
        # so the fairness direction gets an equity-grounded yardstick.
        fairness = "FAIRNESS" in user or "equity" in user.lower() or "gini" in user.lower()
        self._skill_i += 1
        code = _SKILL_CODE
        if self._skill_codes:
            code = self._skill_codes[(self._skill_i - 1) % len(self._skill_codes)]
        return json.dumps({
            "skill_name": f"scripted_skill_{self._skill_i}",
            "objective": ("Equalise driver take-home income (equity)."
                          if fairness else
                          "Serve reachable riders with small detours."),
            "description": ("Routes fares to lift the lowest earners and shrink the "
                            "income spread." if fairness else
                            "Prefers short-pickup, short-ride orders."),
            # The group loop demands both; the legacy hill-climb ignores them.
            "mechanism": f"scripted decision rule {self._skill_i}",
            "differs_from": "weights pickup against ride time differently",
            "fitness_code": _FAIRNESS_FITNESS_CODE if fairness else _FITNESS_CODE,
            "fitness_rationale": ("Low income Gini + lifted income floor capture "
                                  "equity directly." if fairness else
                                  "Service rate and revenue minus service time "
                                  "capture efficient reach."),
            "code": code,
        })


# --------------------------------------------------------------------------- #
# Shared fixtures.                                                             #
# --------------------------------------------------------------------------- #
_STUB_PROFILE = "# TEST ENV PROFILE\nfleet ~120, capacity 4, offpeak, tiny.\n"


def _small_scenarios(n: int, base_seed: int = 0) -> List[Scenario]:
    """A batch of tiny capped-order scenarios so offline rollouts are fast."""
    out: List[Scenario] = []
    for i in range(n):
        out.append(Scenario(
            num_drivers=120, driver_capacity=4, speed_kmh=35.0,
            regime="offpeak", split="train", order_limit=150,
            pref_revenue=round(0.3 + 0.1 * i, 3), seed=base_seed + i,
        ))
    return out


def _two_skill_basis():
    """Compile a fixed {'eff','fair'} skill basis + prompt cards for B2/B3."""
    from pref_dispatch.llm.sandbox import compile_skill

    skills = {
        "eff": compile_skill(_SKILL_CODE, name="eff"),
        "fair": compile_skill(_SKILL_CODE, name="fair"),
    }
    cards = [
        {"skill_name": "eff", "objective": "efficient reach",
         "description": "short-pickup, short-ride orders"},
        {"skill_name": "fair", "objective": "income equity",
         "description": "lift the lowest earners"},
    ]
    return skills, cards


def _two_skill_diff_basis():
    """Basis with GENUINELY different behaviours for the wrong-flip test.

    ``eff`` serves orders (short pickup + ride); ``far`` idles the fleet (high
    no-op). With both active 50/50 the fleet half-serves; a combiner that flips
    the fleet onto ``far`` under an objective must therefore lose reward against
    the rest of the group -- the ranking has something to separate."""
    from pref_dispatch.llm.sandbox import compile_skill

    skills = {
        "eff": compile_skill(_SKILL_CODE, name="eff"),
        "far": compile_skill(_BAD_SKILL_CODE, name="far"),
    }
    cards = [
        {"skill_name": "eff", "objective": "efficient reach",
         "description": "short-pickup, short-ride orders"},
        {"skill_name": "far", "objective": "hold capacity",
         "description": "wait for better orders"},
    ]
    return skills, cards


# --------------------------------------------------------------------------- #
# B0: objective sampler is a key-free distribution of w.                       #
# --------------------------------------------------------------------------- #
def test_objective_sampler() -> None:
    s = ObjectiveSampler(rng=random.Random(0))  # no client => key-free families
    assert "nl" not in s.family_weights, "NL family must be dropped without a client"
    probe = {"assigned_orders": [1], "assigned_party_sizes": {1: 2},
             "assigned_solo_times": {1: 5.0}, "assigned_service_times": {1: 3.0},
             "completed_orders": [], "picked_up_orders": [], "distance_moved": 0.0,
             "time_moved": 0.0, "is_empty_move": False, "is_idle_wait": False,
             "extra_detour_time": 0.0}
    batch = s.sample_batch(20)
    fams = {o.family for o in batch}
    assert fams <= {"raw", "weights", "completion", "pooling"}, fams
    assert "blind" not in fams, "v6 deleted the blind family; it must never be drawn"
    # v10 retired the progressive/nonlinear family: every objective the stack
    # trains and evaluates on is linear in the per-step event terms, so drawing a
    # nonlinear one would put the trainer back on a distribution the gate no
    # longer contains.
    assert "nonlinear" not in fams, \
        f"nonlinear was retired in v10 but was drawn: {fams}"
    # The term-different structural families must be reachable key-free.
    assert fams & {"completion", "pooling"}, \
        f"no structural family drawn in 20 samples: {fams}"
    for o in batch:
        assert not o.is_blind, f"{o.label} has no reward function"
        assert o.spec_text, "every objective must carry a reward spec"
        v = o.w(probe)
        assert isinstance(v, float) and v == v, f"w(probe) not finite: {v}"
    print(f"[B0] objective sampler OK: families={sorted(fams)}, "
          f"{len(batch)} draws, all carry a real reward, key-free")


# --------------------------------------------------------------------------- #
# B1: run_phase1 (1a) evolves one skill per direction, keeps them all.         #
# --------------------------------------------------------------------------- #
def test_phase1_directed() -> None:
    from pref_dispatch.llm.run_phase1 import run_phase1

    client = ScriptedClient()
    directions = (
        "Serve reachable riders with small detours (efficiency).",
        "FAIRNESS DIRECTION: equalise driver take-home income; shrink the Gini.",
    )
    # run_self_invention=False isolates 1a; freeze=False keeps the FS clean.
    res = run_phase1(
        client, _STUB_PROFILE,
        directions=directions,
        run_self_invention=False, freeze=False,
        generations=0, lam=1,             # gen 0 only: fast
        num_drivers=120, order_limit=150,
        regimes=("offpeak",),
        log=lambda *_: None,
    )
    assert res.n_directed == 2, f"expected 2 directed skills, got {res.n_directed}"
    provs = {b.provenance for b in res.directed}
    assert provs == {"directed"}, provs
    for b in res.directed:
        assert b.objective and b.candidate.meta["fitness_rationale"], \
            "directed skill missing NL objective / fitness rationale"
    # The fairness direction must have used the equity-grounded fitness.
    fair = [b for b in res.directed if "equity" in b.objective.lower()
            or "income" in b.objective.lower()]
    assert fair, "no fairness-oriented directed skill produced"
    assert "income_gini" in fair[0].candidate.meta["fitness_code"], \
        "fairness skill's fitness is not grounded in the income-equity KPI"
    print(f"[B1] run_phase1 1a OK: {res.n_directed} directed skills "
          f"(incl. fairness), all kept, NL explanations present")


# --------------------------------------------------------------------------- #
# B1b: the v3 fill->replace loop holds the repository AT max_skills.           #
# --------------------------------------------------------------------------- #
def test_b1_replace_loop() -> None:
    """Exploration must continue past the cap, hold size at N, and stop cleanly.

    Two paths are checked with a repository that starts AT the cap:
      * seeds PROTECTED -> nothing is evictable, so the loop stops immediately on
        ``all_protected`` without shrinking the repository (a required niche such
        as fairness can never be dropped);
      * seeds EVICTABLE with a scripted client that always proposes the SAME
        behaviour -> every proposal loses the redundancy competition, the size
        stays exactly N, and the loop gives up on ``dry_rounds``.
    """
    from pref_dispatch.llm.qd_basis import discover_basis
    from pref_dispatch.skills import EnRouteSkill, RevenueSkill, ServiceSkill

    seeds = [RevenueSkill(), ServiceSkill(), EnRouteSkill()]
    common = dict(generations=0, lam=1, freeze=False, regimes=("offpeak",),
                  num_drivers=120, order_limit=150, log=lambda *_: None)

    protected_res = discover_basis(
        ScriptedClient(), _STUB_PROFILE, seeds, max_skills=3, **common)
    assert protected_res.stop_reason == "all_protected", protected_res.stop_reason
    assert len(protected_res.basis) == 3, len(protected_res.basis)

    dup_res = discover_basis(
        ScriptedClient(), _STUB_PROFILE, seeds, max_skills=3,
        protect_provenance=(), max_dry_rounds=2, max_rounds=6, **common)
    assert dup_res.stop_reason == "dry_rounds", dup_res.stop_reason
    assert len(dup_res.basis) == 3, \
        f"repository must stay at the cap, got {len(dup_res.basis)}"
    assert dup_res.rounds_used >= 2, dup_res.rounds_used
    assert dup_res.n_rejected >= 1, "duplicate proposals were not rejected"
    print(f"[B1b] fill->replace loop OK: held at {len(dup_res.basis)}/3 skills, "
          f"{dup_res.n_rejected} rejected over {dup_res.rounds_used} round(s), "
          f"stop={dup_res.stop_reason}; protected path stop=all_protected")


def test_b1_qd_uses_group_search() -> None:
    """On the sampler path the QD loop must drive the GROUP search, not the
    hill-climb -- and the artifacts it accepts must carry a mechanism.

    ``discover_basis`` keeps two inner searches: the legacy hill-climb when there
    is no sampler (there are no scenario columns to standardise within) and
    ``evolve_skill_group`` when there is. Every other B1 test above runs the
    no-sampler branch, so without this one the whole group path would be untested
    from the QD side -- and a silent fall-through to the hill-climb would still
    produce a plausible-looking repository.
    """
    from pref_dispatch.llm.evolve_skill_group import SkillGroupEval
    from pref_dispatch.llm.qd_basis import discover_basis
    from pref_dispatch.scenario import ScenarioSampler
    from pref_dispatch.skills import RevenueSkill

    # Distinct programs so the group loop's clone-kill leaves >= 2 survivors.
    codes = [_SKILL_CODE,
             _SKILL_CODE.replace("ride - 0.5 * pickup", "ride - 1.5 * pickup"),
             _SKILL_CODE.replace("ride - 0.5 * pickup", "0.25 * ride - pickup")]
    sampler = ScenarioSampler(rng=random.Random(3))
    sig_scenarios = _small_scenarios(2, base_seed=90)
    seen: List[tuple] = []

    res = discover_basis(
        ScriptedClient(skill_codes=codes), _STUB_PROFILE, [RevenueSkill()],
        max_skills=2, protect_provenance=(), max_dry_rounds=1, max_rounds=1,
        sampler=sampler, scenarios_per_round=2, sig_scenarios=sig_scenarios,
        generations=1, mu=2, lam=2, crossover_rate=1.0, fresh_per_round=1,
        checkpoint_fn=lambda r, g, c: seen.append((r, g)),
        freeze=False, log=lambda *_: None,
    )

    evolved = [b for b in res.basis if b.provenance == "evolved"]
    assert evolved, f"the group path produced no skill: {res.stop_reason}"
    b = evolved[0]
    assert isinstance(b.candidate.evaluation, SkillGroupEval), \
        f"QD fell through to the hill-climb: {type(b.candidate.evaluation).__name__}"
    assert b.mechanism, "an accepted skill must carry its mechanism into the basis"
    assert "mechanism" in b.card(), "the diversity card must expose the mechanism"
    ev = b.candidate.evaluation
    assert len(ev.per_scenario_adv) == 2, ev.per_scenario_adv
    assert ev.per_band, "per-band advantages must be recorded"
    # Checkpoints fire per inner generation of the one QD round (0 and 1).
    assert seen == [(1, 0), (1, 1)], seen
    print(f"[B1b] QD -> group search OK: {b.name!r} via {ev.per_band} bands, "
          f"mechanism={b.mechanism!r}, checkpoints={seen}")


def test_b1_directed_uses_group_search() -> None:
    """Step 1a must run the SAME group search as 1b/1c when scenes are available.

    The researcher directions pin the niches the repository is required to cover
    (revenue, waiting, coverage, fairness). Evolving them on the old hill-climb
    while the self-invented skills got (mu+lambda) GRPO would have made exactly
    those required members the weakest programs in the basis -- and nothing in the
    output would have said so, since both searches return a Candidate.
    """
    from pref_dispatch.llm.evolve_skill_group import SkillGroupEval
    from pref_dispatch.llm.run_phase1 import run_phase1
    from pref_dispatch.scenario import ScenarioSampler

    codes = [_SKILL_CODE,
             _SKILL_CODE.replace("ride - 0.5 * pickup", "ride - 1.5 * pickup"),
             _SKILL_CODE.replace("ride - 0.5 * pickup", "0.25 * ride - pickup")]
    seen: List[tuple] = []

    res = run_phase1(
        ScriptedClient(skill_codes=codes), _STUB_PROFILE,
        directions=("Serve reachable riders with small detours (efficiency).",),
        sampler=ScenarioSampler(rng=random.Random(5)),
        scenarios_per_round=2, sig_scenarios=_small_scenarios(2, base_seed=91),
        run_self_invention=False, freeze=False,
        generations=1, mu=2, lam=2, crossover_rate=1.0, fresh_per_round=1,
        checkpoint_fn=lambda stage, i, g, c: seen.append((stage, i, g)),
        log=lambda *_: None,
    )
    assert res.n_directed == 1, res.n_directed
    cand = res.directed[0].candidate
    assert isinstance(cand.evaluation, SkillGroupEval), \
        f"1a fell through to the hill-climb: {type(cand.evaluation).__name__}"
    assert res.directed[0].mechanism, "a directed skill must carry its mechanism"
    assert seen == [("directed", 0, 0), ("directed", 0, 1)], seen
    print(f"[B1] directed -> group search OK: {cand.name!r} via "
          f"{cand.evaluation.per_band} bands, checkpoints={seen}")


def test_b1_banded_windows() -> None:
    """Phase-1 scene batches must be real full hours that SPAN the fleet bands.

    Two properties, both load-bearing for what the selection key measures:
      * no ``order_limit`` -- a capped stream is a 3-minute rush and then an empty
        city, an hour that does not occur in the data;
      * every band represented -- with ``beta = 0`` (2026-08-13) the key is the
        pure mean advantage, so a single-band batch silently turns the round into
        a one-scale contest and a skill that only works at fleet 1200 wins the
        round unopposed.
    """
    from pref_dispatch.llm.batch_pairing import (
        DEFAULT_FLEET_BANDS, BandedWindowSampler, band_label,
    )
    from pref_dispatch.scenario import ScenarioRanges, ScenarioSampler

    ranges = ScenarioRanges(order_limit=None, fleet_dist="loguniform")
    sampler = BandedWindowSampler(
        ScenarioSampler(ranges=ranges, rng=random.Random(0)), ranges=ranges)
    batch = sampler.sample_batch(6, base_seed=100)

    assert len(batch) == 6, len(batch)
    assert all(sc.order_limit is None for sc in batch), \
        [sc.order_limit for sc in batch]
    assert all(sc.window for sc in batch), "every scene must name a real window"
    labels = {band_label(sc) for sc in batch}
    assert len(labels) == len(DEFAULT_FLEET_BANDS), \
        f"batch does not span the bands: {sorted(labels)}"
    # Reproducible: the same sampler state must rebuild the identical grid.
    again = BandedWindowSampler(
        ScenarioSampler(ranges=ranges, rng=random.Random(0)), ranges=ranges,
    ).sample_batch(6, base_seed=100)
    assert [sc.num_drivers for sc in again] == [sc.num_drivers for sc in batch], \
        "banded batch is not reproducible from the same seed"
    print(f"[B1] banded full-hour batch OK: bands={sorted(labels)}, "
          f"fleets={[int(sc.num_drivers) for sc in batch]}, no order cap")


def test_leader_checkpoint() -> None:
    """The per-generation checkpoint must be recoverable, additive, and harmless.

    It exists because a phase writes its artifacts only when it RETURNS: an API
    outage in round 9 of 10 used to lose every program a night of training found.
    Three properties are what make it worth having:
      * the leader's SOURCE is on disk, under ``cache/`` and not in the live
        ``evolved/`` load path (a half-trained program must never join the frozen
        repository mid-run);
      * different searches keep different files while generations of ONE search
        overwrite, so nothing evolved earlier is lost by a later round;
      * a write that fails is logged and swallowed -- a checkpoint that raised
        would kill the run it is there to protect.
    """
    import json
    import os
    import tempfile

    from pref_dispatch.llm.checkpoint import LeaderCheckpoint

    class _Cand:
        def __init__(self, name, code, fit):
            self.name = name
            self.meta = {"code": code, "objective": "maximise revenue"}
            self.evaluation = type("Ev", (), {
                "fitness": fit, "raw_fitness": fit,
                "per_band": {"fleet200-500": 0.1}, "labels": ["s0"],
            })()

    with tempfile.TemporaryDirectory() as d:
        ck = LeaderCheckpoint(1, run="t", root=d, log=lambda *_: None)
        ck("directed", 0, 0, _Cand("a", "def score(): pass", 0.1))
        ck("directed", 0, 1, _Cand("a2", "def score(): return 2", 0.4))
        ck("qd", 3, 0, _Cand("b", "def score(): return 3", 0.2))

        files = sorted(os.listdir(ck.dir))
        assert files == ["directed_0.json", "directed_0.py", "history.jsonl",
                         "qd_3.json", "qd_3.py"], files
        with open(os.path.join(ck.dir, "directed_0.py"), encoding="utf-8") as f:
            src = f.read()
        assert "return 2" in src, "later generation must overwrite its own search"
        assert "CHECKPOINT" in src, "the file must say it is not a frozen artifact"
        with open(os.path.join(ck.dir, "directed_0.json"), encoding="utf-8") as f:
            rec = json.load(f)
        assert rec["name"] == "a2" and rec["evaluation"]["fitness"] == 0.4, rec
        with open(os.path.join(ck.dir, "history.jsonl"), encoding="utf-8") as f:
            hist = [json.loads(ln) for ln in f if ln.strip()]
        assert [h["name"] for h in hist] == ["a", "a2", "b"], hist
        assert d in ck.dir and "evolved" not in ck.dir, ck.dir

        # A candidate the writer cannot serialise must not propagate.
        class _Boom:
            name = "boom"

            @property
            def meta(self):
                raise RuntimeError("no meta")

        lines: List[str] = []
        LeaderCheckpoint(1, run="t2", root=d, log=lines.append)(0, _Boom())
        assert any("FAILED" in ln for ln in lines), lines
    print(f"[B*] leader checkpoint OK: {len(hist)} entries, per-search files, "
          f"latest kept, write failure swallowed")


def test_b1_fill_no_gate() -> None:
    """FILL must not reject on redundancy: below the cap every successfully evolved
    proposal is kept, even a behaviourally identical one; the REPLACE stage then
    rejects the dupe and the loop gives up on dry_rounds.

    With the (wrong) redundancy gate the same inputs would stop at 1 skill; the
    user's v3 rule is "达到规定数量前不替换" -- fill to N first, prune after.
    """
    from pref_dispatch.llm.qd_basis import discover_basis
    from pref_dispatch.skills import RevenueSkill

    res = discover_basis(
        ScriptedClient(), _STUB_PROFILE, [RevenueSkill()],
        max_skills=4, protect_provenance=(), max_dry_rounds=2, max_rounds=8,
        generations=0, lam=1, freeze=False, regimes=("offpeak",),
        num_drivers=120, order_limit=150, log=lambda *_: None,
    )
    assert len(res.basis) == 4, \
        f"fill must reach the cap even with duplicate proposals, got {len(res.basis)}"
    assert res.n_evolved == 3, "all 3 fill proposals must be accepted"
    assert res.stop_reason == "dry_rounds", res.stop_reason
    assert res.n_rejected >= 1, "replace stage must reject the duplicate"
    print(f"[B1b] fill no-gate OK: filled to 4/4 despite duplicate proposals, "
          f"{res.n_rejected} dupe(s) rejected once full, stop={res.stop_reason}")


def test_b1_discard_evicted() -> None:
    """An evicted skill's frozen artifacts must leave the flat basis load path."""
    import glob
    import os
    import tempfile

    from pref_dispatch.llm.evolve import discard_frozen_skill

    with tempfile.TemporaryDirectory() as d:
        py = os.path.join(d, "victim.py")
        meta = os.path.join(d, "victim.meta.json")
        for p, body in ((py, "# frozen skill\n"), (meta, "{}\n")):
            with open(p, "w", encoding="utf-8") as f:
                f.write(body)
        moved = discard_frozen_skill(py)
        # load_basis globs *.meta.json FLAT, so the pair must be gone from there.
        assert glob.glob(os.path.join(d, "*.meta.json")) == [], \
            "evicted meta still visible to load_basis"
        assert os.path.exists(moved) and os.path.exists(
            os.path.join(d, "discarded", "victim.meta.json")), \
            "evicted artifacts were not preserved for audit"
    print("[B1b] evicted artifacts relocated out of the load path (kept for audit)")


# --------------------------------------------------------------------------- #
# B2: evolve_combiner_objectives across a batch of sampled objectives.         #
# --------------------------------------------------------------------------- #
def test_b2_combiner_objectives() -> None:
    import math

    from pref_dispatch.llm.evolve_combiner import evolve_combiner_objectives
    from pref_dispatch.llm.sandbox import validate_combiner

    skills, cards = _two_skill_basis()
    scenarios = _small_scenarios(3, base_seed=0)
    objectives = ObjectiveSampler(rng=random.Random(1)).sample_batch(3)

    best = evolve_combiner_objectives(
        ScriptedClient(), _STUB_PROFILE, skills, cards, scenarios, objectives,
        generations=1, lam=1, log=lambda *_: None,
    )
    assert best.evaluation is not None
    assert math.isfinite(best.evaluation.fitness), best.evaluation.fitness
    ok, why = validate_combiner(best.scorer, tuple(skills))
    assert ok, f"winning combiner failed validation: {why}"
    print(f"[B2] evolve_combiner_objectives OK: fitness={best.evaluation.fitness:.4g} "
          f"over {len(objectives)} objectives x {len(scenarios)} scenes")


# --------------------------------------------------------------------------- #
# B3: evolve_repositioner_objectives across objectives AND fairness strengths. #
# --------------------------------------------------------------------------- #
def test_b3_repositioner_objectives() -> None:
    import math

    from pref_dispatch.combiner import SingleSkillCombiner
    from pref_dispatch.llm.evolve_reposition import evolve_repositioner_objectives
    from pref_dispatch.llm.sandbox import validate_repositioner

    skills, _cards = _two_skill_basis()
    combiner = SingleSkillCombiner("eff")     # frozen dispatch stack for the test
    scenarios = _small_scenarios(2, base_seed=10)
    objectives = ObjectiveSampler(rng=random.Random(2)).sample_batch(2)
    strengths = sample_strengths(random.Random(3), 2)

    best = evolve_repositioner_objectives(
        ScriptedClient(), _STUB_PROFILE, scenarios, objectives, strengths,
        combiner=combiner, skills=skills,
        generations=1, lam=1, log=lambda *_: None,
    )
    assert math.isfinite(best._batch_fitness), best._batch_fitness
    ok, why = validate_repositioner(best.scorer)
    assert ok, f"winning repositioner failed validation: {why}"
    print(f"[B3] evolve_repositioner_objectives OK: mean fitness="
          f"{best._batch_fitness:.4g} over {len(objectives)} objectives x "
          f"strengths={[round(s, 2) for s in strengths]}")


# --------------------------------------------------------------------------- #
# B2b: the objective-blindness DIAGNOSTIC still separates a w-reading combiner  #
#      from a constant one -- and (v6) no longer moves fitness.                 #
# --------------------------------------------------------------------------- #
def test_b2_blindness_discriminates() -> None:
    from pref_dispatch.llm.combiner_adapter import LLMCombiner
    from pref_dispatch.llm.combiner_eval import (
        build_objective_frames,
        evaluate_combiner_objectives,
    )
    from pref_dispatch.llm.objective_sampler import SampledObjective
    from pref_dispatch.llm.sandbox import compile_combiner
    from ride_gym.rewards import (
        CompletionRewardFunction,
        DefaultRewardFunction,
        PoolingRewardFunction,
    )

    skills, _cards = _two_skill_basis()
    scenarios = _small_scenarios(3, base_seed=0)

    def _w(rf):
        return (lambda ev: float(rf(0, ev)))

    # Three term-DIFFERENT objectives: raw (assignment-heavy), completion
    # (pays on drop-off), pooling (pays on seat-filling).
    objs = [
        SampledObjective(
            label="raw", family="raw",
            reward_function=DefaultRewardFunction(
                assignment_bonus=1.0, revenue_coef=0.01, service_time_coef=0.04,
                detour_coef=0.08, empty_move_penalty=0.0, idle_penalty=0.0),
            w=None),
        SampledObjective(
            label="completion", family="completion",
            reward_function=CompletionRewardFunction(
                completion_bonus=2.0, assignment_bonus=0.2, detour_coef=0.02),
            w=None),
        SampledObjective(
            label="pooling", family="pooling",
            reward_function=PoolingRewardFunction(
                solo_bonus=0.2, party_bonus=1.5, detour_coef=0.05),
            w=None),
    ]
    for o in objs:
        o.w = _w(o.reward_function)
    frames = build_objective_frames(
        skills, scenarios, [o.reward_function for o in objs])

    blind = LLMCombiner(compile_combiner(_BLIND_COMBINER_CODE), ("eff", "fair"))
    diff = LLMCombiner(compile_combiner(_DIFF_COMBINER_CODE), ("eff", "fair"))
    ev_b = evaluate_combiner_objectives(blind, skills, scenarios, objs, frames)
    ev_d = evaluate_combiner_objectives(diff, skills, scenarios, objs, frames)

    assert ev_b.objective_blindness > 0.9, \
        f"constant combiner must read as objective-blind, got {ev_b.objective_blindness:.3f}"
    assert ev_d.objective_blindness < 0.5, \
        f"flipping combiner must read as objective-aware, got {ev_d.objective_blindness:.3f}"
    # Both skills are the SAME code, so the raw batch score is EXACTLY equal --
    # and since v6 deleted the blindness PENALTY, the fitness must be equal too:
    # the number is a report, not a lever. (v5 asserted the opposite here.)
    assert ev_b.raw_fitness == ev_d.raw_fitness
    assert ev_b.fitness == ev_d.fitness, \
        f"blindness must not move fitness in v6: {ev_b.fitness:.4g} vs {ev_d.fitness:.4g}"
    assert ev_b.fitness == ev_b.raw_fitness - 0.5 * ev_b.fallback_rate, \
        "fallback penalty must be the only subtraction left"
    print(f"[B2b] blindness DIAGNOSTIC still separates: blind={ev_b.objective_blindness:.2f} "
          f"vs adapting={ev_d.objective_blindness:.2f}, and it no longer moves "
          f"fitness (both {ev_b.fitness:.4g})")


# --------------------------------------------------------------------------- #
# B2d: what the deleted anti-harm penalty was for, done by the group instead.   #
#      A combiner that steers the fleet the WRONG way under an objective must   #
#      lose to one that steers it the right way -- v5 bought that with a second #
#      w=None rollout per pair plus a penalty term; v6 gets it for free from    #
#      ranking the two candidates against each other on the same objective.     #
# --------------------------------------------------------------------------- #
def test_b2_wrong_flip_loses_in_group() -> None:
    from pref_dispatch.llm.combiner_adapter import LLMCombiner
    from pref_dispatch.llm.combiner_eval import evaluate_combiner_group
    from pref_dispatch.llm.objective_sampler import SampledObjective
    from pref_dispatch.llm.sandbox import compile_combiner
    from ride_gym.rewards import (
        CompletionRewardFunction,
        PoolingRewardFunction,
    )

    skills, _cards = _two_skill_diff_basis()   # eff serves, far idles
    scenarios = _small_scenarios(2, base_seed=20)

    def _w(rf):
        return (lambda ev: float(rf(0, ev)))

    objs = [
        SampledObjective(
            label="completion", family="completion",
            reward_function=CompletionRewardFunction(
                completion_bonus=2.0, assignment_bonus=0.2, detour_coef=0.02),
            w=None),
        SampledObjective(
            label="pooling", family="pooling",
            reward_function=PoolingRewardFunction(
                solo_bonus=0.2, party_bonus=1.5, detour_coef=0.05),
            w=None),
    ]
    for o in objs:
        o.w = _w(o.reward_function)

    wrong = LLMCombiner(compile_combiner(_WRONG_FLIP_COMBINER_CODE), ("eff", "far"))
    right = LLMCombiner(compile_combiner(_RIGHT_FLIP_COMBINER_CODE), ("eff", "far"))
    ev_w, ev_r = evaluate_combiner_group(
        [wrong, right], skills, scenarios, objs)

    # Both are equally objective-aware (the argmax moves for both), so the only
    # thing separating them is WHERE they move it -- exactly what the anti-harm
    # term used to measure, now read straight off the group advantage.
    assert ev_r.raw_fitness > ev_w.raw_fitness, \
        (f"the group advantage must sink the wrong-flip combiner: "
         f"wrong {ev_w.raw_fitness:.3f} vs right {ev_r.raw_fitness:.3f}")
    # Since 2026-08-12 the group path charges NOTHING on top of the delta: a break
    # runs the equal blend, which IS the subtracted baseline, so it prices itself.
    assert ev_w.fitness == ev_w.raw_fitness, \
        "the group fitness must have no penalty term left"
    print(f"[B2d] the group advantage replaces the anti-harm penalty: wrong-flip "
          f"scores {ev_w.raw_fitness:+.2f}, right-flip {ev_r.raw_fitness:+.2f} "
          f"(one rollout per pair, not two)")



# --------------------------------------------------------------------------- #
# B2e: GRPO-style GROUP-RELATIVE fitness (the training redesign). The fitness
#      is now the per-objective percentile inside the round's OWN pooled group
#      ({candidates, frozen skills, each candidate's w=None baseline}), which
#      fixes the two failures of the old per-scenario min-max ruler: objective
#      INTERNAL scale (a 2x weight must rank identically) and ruler saturation
#      (no discrimination between candidates).
# --------------------------------------------------------------------------- #
def _two_obj_batch(base_seed: int):
    """A completion + pooling objective pair over a 2-scenario batch (B2e uses
    the SAME serving-vs-idling basis as B2d so a wrong flip has real teeth)."""
    from pref_dispatch.llm.objective_sampler import SampledObjective
    from ride_gym.rewards import (
        CompletionRewardFunction,
        PoolingRewardFunction,
    )

    def _w(rf):
        return (lambda ev: float(rf(0, ev)))

    objs = [
        SampledObjective(
            label="completion", family="completion",
            reward_function=CompletionRewardFunction(
                completion_bonus=2.0, assignment_bonus=0.2, detour_coef=0.02),
            w=None),
        SampledObjective(
            label="pooling", family="pooling",
            reward_function=PoolingRewardFunction(
                solo_bonus=0.2, party_bonus=1.5, detour_coef=0.05),
            w=None),
    ]
    for o in objs:
        o.w = _w(o.reward_function)
    scenarios = _small_scenarios(2, base_seed=base_seed)
    return scenarios, objs


def test_b2e_group_relative_scale_invariance() -> None:
    """Two objectives differing ONLY by a 2x weight must score identically.

    This is the INTERNAL-scale failure of the old fitness: per-scenario min-max
    normalisation cannot see that a 2x objective is the SAME difficulty, so the
    old signal could not tell which method is good/bad for an objective. The
    standardised advantage is scale-free by construction: doubling the reward
    function doubles every member of the group, so the mean and the spread double
    with it and ``(r-mean)/std`` is unchanged.

    The two pairs run the SAME scene, so the objective's scale is the only thing
    that differs. (Before 2026-08-10 this fixture used two DIFFERENT scenes and
    still asserted exact equality -- which passed only because a percentile over a
    4-member group is coarse enough to collide. It was not testing the claim.)"""
    from pref_dispatch.llm.combiner_adapter import LLMCombiner
    from pref_dispatch.llm.combiner_eval import evaluate_combiner_group
    from pref_dispatch.llm.objective_sampler import SampledObjective
    from pref_dispatch.llm.sandbox import compile_combiner
    from ride_gym.rewards import DefaultRewardFunction

    skills, _cards = _two_skill_diff_basis()
    one = _small_scenarios(1, base_seed=40)[0]
    scenarios = [one, one]

    def _w(rf):
        return (lambda ev: float(rf(0, ev)))

    kw = dict(assignment_bonus=1.0, revenue_coef=0.01, service_time_coef=0.04,
              detour_coef=0.08, empty_move_penalty=0.0, idle_penalty=0.0)
    objs = [
        SampledObjective(label="a", family="raw",
                         reward_function=DefaultRewardFunction(**kw), w=None),
        SampledObjective(label="a2x", family="raw",
                         reward_function=DefaultRewardFunction(
                             **{k: 2.0 * v for k, v in kw.items()}), w=None),
    ]
    for o in objs:
        o.w = _w(o.reward_function)

    bal = LLMCombiner(compile_combiner(_BALANCED_COMBINER_CODE), ("eff", "far"))
    diff = LLMCombiner(compile_combiner(_RIGHT_FLIP_COMBINER_CODE), ("eff", "far"))
    evs = evaluate_combiner_group([bal, diff], skills, scenarios, objs)
    for ev in evs:
        assert abs(ev.per_pref[0] - ev.per_pref[1]) < 1e-9, \
            f"2x-scaled objective must score identically: {ev.per_pref}"
        assert ev.raw_fitness == (ev.per_pref[0] + ev.per_pref[1]) / 2
    print(f"[B2e] scale-invariance OK: pair-0 == pair-1 advantages for both "
          f"candidates ({[round(ev.raw_fitness, 3) for ev in evs]}) despite a 2x "
          f"objective")


def test_b2e_group_relative_discriminates() -> None:
    """Good > middling > bad must ORDER by group advantage, with a real spread.

    This is the discrimination the old ruler saturated away: candidates are
    mixtures of skills that beat the single-skill band, so nearly everyone scored
    near 1.0 and the arbitrary gen-0 seed never lost a generation. Standardising
    inside the round's own group keeps the differences visible."""
    from pref_dispatch.llm.combiner_adapter import LLMCombiner
    from pref_dispatch.llm.combiner_eval import evaluate_combiner_group
    from pref_dispatch.llm.sandbox import compile_combiner

    skills, _cards = _two_skill_diff_basis()
    scenarios, objs = _two_obj_batch(base_seed=60)

    good = LLMCombiner(compile_combiner(_RIGHT_FLIP_COMBINER_CODE), ("eff", "far"))
    mid = LLMCombiner(compile_combiner(_BALANCED_COMBINER_CODE), ("eff", "far"))
    bad = LLMCombiner(compile_combiner(_FAR_LEAN_COMBINER_CODE), ("eff", "far"))
    evs = evaluate_combiner_group([good, mid, bad], skills, scenarios, objs)

    assert evs[0].raw_fitness > evs[1].raw_fitness > evs[2].raw_fitness, \
        f"quality order must survive the group: " \
        f"good={evs[0].raw_fitness:.3f} mid={evs[1].raw_fitness:.3f} bad={evs[2].raw_fitness:.3f}"
    assert evs[0].raw_fitness - evs[2].raw_fitness > 0.2, \
        f"group must not saturate: spread {evs[0].raw_fitness - evs[2].raw_fitness:.3f}"
    print(f"[B2e] discrimination OK: good {evs[0].raw_fitness:.2f} > "
          f"mid {evs[1].raw_fitness:.2f} > bad {evs[2].raw_fitness:.2f}")


def test_b2e_skill_anchor() -> None:
    """Single-skill candidates (the frozen skills as combiners) behave as the
    group's anchors: the serving skill's candidate scores above the idling one on
    the batch, and both advantages stay inside the +/-3 clip."""
    from pref_dispatch.combiner import SingleSkillCombiner
    from pref_dispatch.llm.combiner_eval import evaluate_combiner_group
    from pref_dispatch.llm.group_fitness import Z_CLIP

    skills, _cards = _two_skill_diff_basis()
    scenarios, objs = _two_obj_batch(base_seed=100)

    ev_eff, ev_far = evaluate_combiner_group(
        [SingleSkillCombiner("eff"), SingleSkillCombiner("far")],
        skills, scenarios, objs,
    )
    assert ev_eff.raw_fitness > ev_far.raw_fitness, \
        f"serving anchor must beat the idling one: {ev_eff.raw_fitness:.3f} vs {ev_far.raw_fitness:.3f}"
    assert -Z_CLIP <= ev_far.raw_fitness < ev_eff.raw_fitness <= Z_CLIP
    # Since the delta fitness the zero crossing is the EQUAL BLEND, not the group
    # mean. In THIS basis the blend is degenerate: ``far`` returns noop_score 10.0
    # against ``eff``'s 0.2, so a 50/50 blend still lets the no-op win every
    # dispatch and serves nothing -- the equal blend and pure idling earn the same
    # 0.0. So the idling anchor sits EXACTLY on the baseline (it really is worth
    # what not choosing is worth) and the serving one is strictly above it. Under
    # the old centred advantage these two signs were merely each other's mirror --
    # any two members straddle their own mean -- and said nothing about either
    # one's absolute worth. See test_b2e_delta_sign_is_absolute for a basis where
    # the blend is NOT degenerate and both anchors come out negative.
    assert ev_eff.per_family["completion"] > 0.0, ev_eff.per_family
    assert ev_far.per_family["completion"] == 0.0, ev_far.per_family
    print(f"[B2e] skill anchor OK: eff {ev_eff.raw_fitness:+.2f} > far "
          f"{ev_far.raw_fitness:+.2f}; per-family {ev_eff.per_family}")


def test_b2e_crash_is_the_baseline() -> None:
    """A combiner that crashes on every driver must score EXACTLY 0.0 -- no
    penalty coefficient involved.

    Before 2026-08-12 a crash returned ``{skill_names[0]: 1.0}``: a WORKING
    single-skill policy. A program that never ran silently inherited that skill's
    result, and ``fallback_penalty`` existed to charge back the borrowed credit --
    a hyperparameter buying back an accident of dict ordering. Now a crash returns
    the EQUAL BLEND, which is exactly the policy the fitness subtracts, so the
    delta is 0 on every pair and the failure prices itself.

    Two things are asserted together, because either alone can pass by luck: the
    program really did break on ~every decision, and its fitness is 0.0 to the bit
    (not merely small)."""
    from pref_dispatch.llm.combiner_adapter import LLMCombiner
    from pref_dispatch.llm.combiner_eval import evaluate_combiner_group
    from pref_dispatch.llm.sandbox import compile_combiner

    skills, _cards = _two_skill_diff_basis()
    scenarios, objs = _two_obj_batch(base_seed=120)

    crash = LLMCombiner(compile_combiner(_CRASHING_COMBINER_CODE), ("eff", "far"))
    # The fixture is written to survive the sandbox's synthetic probes and only
    # start raising after 50 calls -- otherwise it would never compile. Those 50
    # calls would land on the first 50 REAL drivers, who would run a WORKING
    # single-skill policy (measured: +0.165 on pair 0), and the delta would not be
    # exactly 0. Validation is done by now, so trip it here: the claim under test
    # is what a program that breaks on EVERY decision scores.
    crash.scorer.skill_scores.__globals__["_SEEN"][0] = 10 ** 9
    ev = evaluate_combiner_group([crash], skills, scenarios, objs)[0]
    assert ev.fallback_rate == 1.0, \
        f"crashing combiner must fall back on EVERY decision: {ev.fallback_rate:.3f}"
    assert ev.fitness == ev.raw_fitness, \
        "no penalty term may be left on the group path"
    assert ev.raw_fitness == 0.0, \
        (f"a crash runs the equal blend = the baseline, so its delta must be "
         f"exactly 0, got {ev.raw_fitness!r}")
    assert all(v == 0.0 for v in ev.per_pref.values()), \
        f"every pair must be exactly 0, got {ev.per_pref}"
    print(f"[B2e] crash-is-the-baseline OK: fallback_rate={ev.fallback_rate:.2f}, "
          f"fitness {ev.fitness:.3f} (exactly the do-nothing baseline, no penalty)")


def test_b2e_delta_sign_is_absolute() -> None:
    """The sign says something on its own -- and in particular EVERY candidate in a
    round is allowed to be negative AT ONCE.

    That is the statement the centred ``(reward - group mean) / std`` structurally
    could not make: any two members straddle their own mean, so one of them came out
    positive no matter how bad both were. "Worth nothing" and "middle of a useless
    field" printed the same 0.00. Against a fixed baseline they separate.

    The basis here is two skills that BOTH genuinely serve -- ``quick`` prefers short
    pickups, ``slow`` prefers long rides -- so the equal blend is a real policy and
    not the do-nothing floor it degenerates to in :func:`_two_skill_diff_basis`
    (where ``far``'s noop_score 10.0 wins every dispatch even at 50/50). It also
    happens to BEAT both of them: mixing two scorers that rank orders differently
    serves more here than either alone. So committing the whole fleet to either
    single skill is worth less than not choosing, and both must read below zero.

    Neither program can crash, which is the other half of the point: this is a
    reliable candidate scoring negative, not a failure being punished."""
    from pref_dispatch.combiner import SingleSkillCombiner
    from pref_dispatch.llm.combiner_eval import evaluate_combiner_group
    from pref_dispatch.llm.sandbox import compile_skill

    skills = {"quick": compile_skill(_SKILL_CODE, name="quick"),
              "slow": compile_skill(_SLOW_SKILL_CODE, name="slow")}
    scenarios, objs = _two_obj_batch(base_seed=100)

    ev_q, ev_s = evaluate_combiner_group(
        [SingleSkillCombiner("quick"), SingleSkillCombiner("slow")],
        skills, scenarios, objs,
    )
    assert ev_q.fallback_rate == 0.0 and ev_s.fallback_rate == 0.0, \
        (f"both must be RELIABLE -- the point is that WORKING programs score below "
         f"0: {ev_q.fallback_rate:.3f}, {ev_s.fallback_rate:.3f}")
    assert ev_q.raw_fitness < 0.0 and ev_s.raw_fitness < 0.0, \
        (f"both single skills lose to the equal blend on this basis, so BOTH must "
         f"be negative: quick {ev_q.raw_fitness:+.3f}, slow {ev_s.raw_fitness:+.3f}")
    print(f"[B2e] absolute sign OK: BOTH candidates negative in the same round "
          f"(quick {ev_q.raw_fitness:+.2f}, slow {ev_s.raw_fitness:+.2f}) -- "
          f"unreachable under a group-centred advantage")


# --------------------------------------------------------------------------- #
# B2c: the objective sampler stratifies, so a small batch cannot lose a family. #
# --------------------------------------------------------------------------- #
def test_b2_stratified_batch() -> None:
    import pref_dispatch.llm.objective_sampler as os_mod
    from pref_dispatch.llm.objective_sampler import SampledObjective

    s = ObjectiveSampler(rng=random.Random(7))          # no client: key-free
    cnt = s._stratified_counts(8)
    assert cnt["structural"] >= 2, cnt
    assert "blind" not in cnt, f"v6 deleted the blind family: {cnt}"
    assert cnt.get("nl", 0) == 0, "NL must not be planned without a client"
    batch = s.sample_batch(8)                            # the real draw path
    fams = [o.family for o in batch]
    n_struct = sum(1 for f in fams if f in ("completion", "pooling"))
    assert n_struct >= 2, f"stratified batch lost structural coverage: {fams}"
    assert "blind" not in fams, f"blind family must never be drawn: {fams}"
    # With a client the NL family is guaranteed in the plan (no LLM call here).
    s_client = ObjectiveSampler(client=object(), rng=random.Random(7))
    assert s_client._stratified_counts(8)["nl"] == 1
    # Call-site arity: a client-present batch must route exactly one draw to the
    # NL family WITHOUT a TypeError (the client/log args are threaded to the
    # authoring helpers). Patch the helpers so no network/key is involved.
    seen: Dict[str, int] = {}
    fake_client = object()

    def _fake_nl(rng, client, briefs, temperature, log):
        seen["nl"] = seen.get("nl", 0) + 1
        assert client is fake_client, "client must reach the NL authoring helper"
        return SampledObjective(label="nl", family="nl", reward_function=None, w=None)

    def _fake_weights(rng, client, temperature, log):
        seen["weights"] = seen.get("weights", 0) + 1
        return SampledObjective(label="w", family="weights", reward_function=None, w=None)

    orig_nl, orig_w = os_mod._nl_objective, os_mod._weights_objective
    os_mod._nl_objective, os_mod._weights_objective = _fake_nl, _fake_weights
    try:
        batch_c = ObjectiveSampler(client=fake_client, rng=random.Random(7)).sample_batch(8)
    finally:
        os_mod._nl_objective, os_mod._weights_objective = orig_nl, orig_w
    assert len(batch_c) == 8 and seen["nl"] == 1, (len(batch_c), seen)
    print(f"[B2c] stratified batch OK: families={fams}; client-present NL draw OK "
          f"(nl={seen['nl']}, weights={seen['weights']})")


# --------------------------------------------------------------------------- #
# B2f: the selection key = PURE GRPO mean advantage (beta = 0, 2026-08-13).
#      Pure arithmetic on selection_score(ev, beta) = ev.fitness + beta x
#      weakest-family standing: no rollouts, no env -- the point is the SELECTION
#      KEY, not the group.
# --------------------------------------------------------------------------- #
def _mk_eval(fitness: float, per_family: Dict[str, float]) -> "CombinerEval":
    from pref_dispatch.llm.combiner_eval import CombinerEval
    return CombinerEval(fitness=fitness, raw_fitness=fitness, fallback_rate=0.0,
                        per_family=dict(per_family))


def test_b2f_selection_score_bias() -> None:
    """Weak-family bias: near-equal mean -> higher floor wins; no mediocrity hole;
    a crasher stays behind regardless of its min.

    The per-family dicts below are the v1 champion's REAL measured shape, kept
    verbatim as the historical record this rule was written against. They still
    name ``nonlinear``, a family retired in v10 -- that is deliberate and not a
    stale reference: ``selection_score`` is family-AGNOSTIC (it reads the mean and
    the min over whatever keys it is handed), so these fixtures exercise the
    arithmetic without asserting anything about which families the sampler draws
    today. :func:`test_objective_sampler` is what pins the live family set.
    """
    from pref_dispatch.llm.evolve_combiner import selection_score

    # beta = 0 (user requirement 2026-08-13): the selection key is EXACTLY the
    # pure GRPO mean advantage; no hand-set constant weights any family. The
    # v1 champion's fixture is kept as the historical record the old 0.15 rule
    # was written against, but the assertions below pin the NEW semantics:
    # 1) the key equals fitness exactly; 2) a specialist with the SAME mean
    # outranks an all-rounder with the same mean (the per-family elite SLOT --
    # not the key -- is what keeps a hard-family specialist alive, and a
    # specialist with equal mean is strictly better since it has no weak floor).
    champion = _mk_eval(0.5126, {
        "pooling": 0.54, "completion": 0.81, "nl": 0.45,
        "nonlinear": 0.38, "raw": 0.31, "weights": 0.53,
    })
    # A flat mediocre candidate: constant blend, every family ~equal to its
    # mean. With beta=0 its key is its mean, below the champion's -- no hole.
    mediocre = _mk_eval(0.45, {k: 0.45 for k in champion.per_family})
    # Equal means, DIFFERENT floors: with beta=0 both keys are equal -- the
    # specialist is not penalised for having a strong floor and not rewarded
    # beyond it either; it survives via the reserved per-family slot instead.
    allrounder = _mk_eval(0.55, {"raw": 0.80, "completion": 0.80,
                                 "nonlinear": 0.30, "pooling": 0.80})
    specialist = _mk_eval(0.55, {"raw": 0.80, "completion": 0.80,
                                 "nonlinear": 0.80, "pooling": 0.80})
    # A crasher: fallback-penalised fitness far below the group, high min or not.
    crasher = _mk_eval(0.20, {"raw": 0.60, "completion": 0.60,
                              "nonlinear": 0.60, "pooling": 0.60})

    s_champ, s_med = selection_score(champion), selection_score(mediocre)
    assert s_med < s_champ, \
        f"mediocrity hole: flat candidate {s_med:.3f} >= champion {s_champ:.3f}"
    s_ar, s_sp = selection_score(allrounder), selection_score(specialist)
    assert abs(s_sp - s_ar) < 1e-9, \
        f"beta=0: equal-mean specialist and all-rounder must tie: {s_sp:.3f} vs {s_ar:.3f}"
    assert s_ar == allrounder.fitness, \
        f"beta=0: selection must equal fitness exactly, got {s_ar:.3f} != {allrounder.fitness:.3f}"
    assert selection_score(crasher) < s_champ, \
        "crasher must stay behind the champion"
    assert s_sp > s_champ, "an equal-mean specialist must still beat the v1 champion"
    print(f"[B2f] selection OK (beta=0): champion {s_champ:.3f} > flat-mediocre "
          f"{s_med:.3f} (no hole); specialist {s_sp:.3f} == all-rounder {s_ar:.3f} "
          f"(key is pure mean); crasher {selection_score(crasher):.3f} behind")


def test_b2f_accept_rule() -> None:
    """The accept rule is ``selection(new) >= selection(incumbent)``, and with
    ``beta = 0`` (2026-08-13) selection IS the pure GRPO mean advantage. So a
    candidate ACCEPTS iff its MEAN is at least the incumbent's -- a weak-family
    fix that dips the mean must now REJECT on the key (the family elite SLOT, not
    the key, is what keeps a hard-family specialist alive), and a flat mediocre
    candidate with a lower mean rejects too."""
    from pref_dispatch.llm.evolve_combiner import selection_score

    # Incumbent = the v1 champion (fitness 0.5126).
    incumbent = _mk_eval(0.5126, {
        "pooling": 0.54, "completion": 0.81, "nl": 0.45,
        "nonlinear": 0.38, "raw": 0.31, "weights": 0.53,
    })
    s_inc = selection_score(incumbent)
    assert s_inc == incumbent.fitness, \
        f"beta=0: selection must equal fitness, got {s_inc:.3f}"

    # The weak-family fix: mean dips 0.5126 -> 0.51. With beta=0 its key is BELOW
    # the incumbent's, so it rejects on the key (and survives only via the family
    # elite slot if it is the best on some family).
    weak_fix = _mk_eval(0.51, {
        "pooling": 0.53, "completion": 0.79, "nl": 0.45,
        "nonlinear": 0.45, "raw": 0.46, "weights": 0.52,
    })
    s_fix = selection_score(weak_fix)
    assert s_fix < s_inc, \
        f"beta=0: lower-mean weak-family fix must reject on the key: {s_fix:.3f} vs {s_inc:.3f}"

    # The flat mediocre: lower mean, must reject.
    mediocre = _mk_eval(0.48, {k: 0.48 for k in incumbent.per_family})
    s_med = selection_score(mediocre)
    assert s_med < s_inc, \
        f"flat mediocre must reject: {s_med:.3f} vs incumbent {s_inc:.3f}"
    print(f"[B2f] accept rule OK (beta=0): key==mean ({s_inc:.3f}); "
          f"lower-mean weak-family fix {s_fix:.3f} < {s_inc:.3f} reject on key; "
          f"flat-mediocre {s_med:.3f} < {s_inc:.3f} reject")


# --------------------------------------------------------------------------- #
# B2h (v6): (mu+lambda)-ES -- family elite slots, rotating scenes, crossover,   #
# and the single-skill yardstick that left the selection group.                 #
# --------------------------------------------------------------------------- #
def _fake_candidate(name: str, fitness: float, per_family: Dict[str, float]):
    """A CombinerCandidate shell carrying only a name and an evaluation."""
    from pref_dispatch.llm.evolve_combiner import CombinerCandidate
    c = CombinerCandidate(meta={"combiner_name": name, "code": "", "strategy": ""},
                          scorer=None, skill_names=())
    c.evaluation = _mk_eval(fitness, per_family)
    return c


class _Obj:
    """Minimal stand-in for a SampledObjective (only ``family`` is read here)."""

    def __init__(self, family: str):
        self.family = family


def test_b2h_family_elite_slot() -> None:
    """The reserved per-family slot must save the ONLY program that is good at a
    hard family, even when four blander programs outrank it overall.

    This is the culling that plain top-mu does and v6 must not: the completion
    specialist below has the worst mean of the five, so top-4 drops it -- and with
    it the only mechanism crossover could have harvested for the gate's
    scarce-fleet completion cells."""
    from pref_dispatch.llm.evolve_combiner import select_survivors

    objs = [_Obj("raw"), _Obj("completion"), _Obj("pooling")]
    fams = ("raw", "completion", "pooling")
    pool = [_fake_candidate(f"bland{i}", 0.60 - 0.01 * i,
                            {f: 0.60 - 0.01 * i for f in fams})
            for i in range(4)]
    # Mean 0.45 (last place), but 0.95 on completion -- nobody else is close.
    spec = _fake_candidate("completion_specialist", 0.45,
                           {"raw": 0.20, "completion": 0.95, "pooling": 0.20})
    pool.append(spec)

    plain = sorted(pool, key=lambda c: c.evaluation.fitness, reverse=True)[:4]
    assert spec not in plain, "fixture broken: specialist must lose the plain top-mu cut"

    keep = select_survivors(pool, objs, mu=4)
    names = [c.name for c in keep]
    assert "completion_specialist" in names, \
        f"family elite slot did not save the specialist: {names}"
    assert len(keep) <= 4 + len(fams), f"archive blew past mu + n_families: {names}"
    # And the slot must not bloat the archive when the elites are already inside
    # the top mu: same pool, mu=5 -> exactly 5 survivors, no duplicate rows.
    keep5 = select_survivors(pool, objs, mu=5)
    assert len(keep5) == 5 and len({id(c) for c in keep5}) == 5, \
        f"elite slot duplicated an already-surviving program: {[c.name for c in keep5]}"
    print(f"[B2h] family elite slot OK: plain top-4 = {[c.name for c in plain]} drops "
          f"the specialist (mean 0.45); with elites -> {names}")


def test_b2h_mu_lambda_rotating_scenes() -> None:
    """The (mu+lambda) loop must draw a FRESH batch every round, re-roll the
    parents on it, and reach the crossover operator when two survive.

    Guards three v6 properties at once: batch_fn is consulted exactly once per
    round (0..G), the scenes it returns actually differ between rounds, and every
    surviving parent carries an evaluation measured on the LAST round's grid (not
    the grid it was admitted on).

    The client is scripted with FOUR behaviourally distinct programs over the
    eff/far basis, because both are required for the loop to reach crossover at
    all: with one fixed program (or with the eff/fair basis, whose two skills are
    compiled from the SAME source and therefore score identically no matter how
    they are weighted) every rollout row of a round is byte-identical, the
    clone-kill eliminates all but the first candidate, and ``_offspring``'s
    ``len(survivors) >= 2`` crossover branch is never taken."""
    import math

    from pref_dispatch.llm.evolve_combiner import evolve_combiner_objectives

    skills, cards = _two_skill_diff_basis()
    client = ScriptedClient(combiner_codes=[
        _BALANCED_COMBINER_CODE,     # serves half the fleet
        _FAR_LEAN_COMBINER_CODE,     # idles almost always
        _RIGHT_FLIP_COMBINER_CODE,   # serves always
        _WRONG_FLIP_COMBINER_CODE,   # idles under completion-shaped w
    ])
    seen_rounds = []

    def batch_fn(r: int):
        seen_rounds.append(r)
        scs = _small_scenarios(2, base_seed=100 * (r + 1))
        objs = ObjectiveSampler(rng=random.Random(50 + r)).sample_batch(2)
        return scs, objs

    scs0, objs0 = batch_fn(-1)          # sizing batch, not one of the rounds
    seen_rounds.clear()

    ckpts: list = []

    lines: list = []
    best = evolve_combiner_objectives(
        client, _STUB_PROFILE, skills, cards, scs0, objs0,
        batch_fn=batch_fn, generations=2, mu=2, lam=2,
        crossover_rate=1.0, rng=random.Random(0), log=lines.append,
        checkpoint_fn=lambda g, c: ckpts.append((g, c.name)),
    )
    assert seen_rounds == [0, 1, 2], f"batch_fn not called once per round: {seen_rounds}"
    # A run that dies in round 2 must still leave round 1's leader on disk, so the
    # leader is handed out after EVERY round including the gen-0 archive.
    assert [g for g, _ in ckpts] == [0, 1, 2], \
        f"checkpoint_fn not fired once per round: {ckpts}"
    assert ckpts[-1][1] == best.name, \
        f"last checkpoint {ckpts[-1][1]!r} is not the returned winner {best.name!r}"
    assert best.evaluation is not None and math.isfinite(best.evaluation.fitness)
    # crossover_rate=1.0 with mu=2 survivors: every gen-1+ child is a cross, and
    # the parents must be RE-ROLLED next to it (they reappear tagged [parent]).
    crosses = [ln for ln in lines if "crossover(" in ln]
    parents = [ln for ln in lines if "[parent]" in ln]
    assert crosses, f"crossover operator never reached:\n{chr(10).join(lines)}"
    assert parents, f"parents were not re-rolled with the offspring:\n{chr(10).join(lines)}"
    # The scenes really rotated: two rounds must not share an env seed set.
    a = {s.seed for s in _small_scenarios(2, base_seed=100)}
    b = {s.seed for s in _small_scenarios(2, base_seed=200)}
    assert a != b, "fixture broken: rotating batches produced identical seeds"
    print(f"[B2h] (mu=2+lambda=2) rotating-scene loop OK: rounds {seen_rounds}, "
          f"{len(crosses)} crossover child/children, {len(parents)} parent re-rolls, "
          f"winner {best.name!r} fitness {best.evaluation.fitness:.3f}")


def test_b2h_patience_and_runoff() -> None:
    """Adaptive stop + runoff final (2026-08-13).

    Three properties:
      * ``patience=K``: the loop stops early once the SAME code has led K
        consecutive rounds -- fewer batch_fn calls than the full-length run.
      * ``runoff=True``: distinct round leaders are re-rolled together on ONE
        extra fresh batch after the loop, and the returned champion carries an
        evaluation measured on THAT batch (the runoff round index).
      * runoff with a single distinct leader is skipped (no wasted rollouts).
    """
    from pref_dispatch.llm.evolve_combiner import evolve_combiner_objectives

    skills, cards = _two_skill_diff_basis()

    # --- patience: a client that always returns the SAME program -> the leader
    # code never changes, so patience=2 must stop the run before generations=6.
    client_same = ScriptedClient(combiner_codes=[_BALANCED_COMBINER_CODE])
    seen_rounds: list = []

    def batch_fn(r: int):
        seen_rounds.append(r)
        scs = _small_scenarios(2, base_seed=100 * (r + 2))
        objs = ObjectiveSampler(rng=random.Random(60 + r)).sample_batch(2)
        return scs, objs

    scs0, objs0 = batch_fn(-1)
    seen_rounds.clear()
    lines: list = []
    best = evolve_combiner_objectives(
        client_same, _STUB_PROFILE, skills, cards, scs0, objs0,
        batch_fn=batch_fn, generations=6, mu=2, lam=2,
        crossover_rate=0.0, rng=random.Random(0), log=lines.append,
        patience=2, runoff=False,
    )
    stopped = [ln for ln in lines if ln.startswith("[stop]")]
    assert stopped, f"patience never fired:\n{chr(10).join(lines)}"
    assert max(seen_rounds) < 6, \
        f"patience=2 should stop before the cap; rounds ran: {seen_rounds}"
    assert best.evaluation is not None

    # --- runoff: four behaviourally distinct programs make leaders change, so
    # several distinct leaders accumulate; the winner must then be evaluated on
    # the runoff batch (one more batch_fn call than the last round).
    client_many = ScriptedClient(combiner_codes=[
        _BALANCED_COMBINER_CODE, _FAR_LEAN_COMBINER_CODE,
        _RIGHT_FLIP_COMBINER_CODE, _WRONG_FLIP_COMBINER_CODE,
    ])
    seen_rounds.clear()
    lines2: list = []
    best2 = evolve_combiner_objectives(
        client_many, _STUB_PROFILE, skills, cards, scs0, objs0,
        batch_fn=batch_fn, generations=2, mu=2, lam=2,
        crossover_rate=1.0, rng=random.Random(0), log=lines2.append,
        patience=0, runoff=True,
    )
    runoff_lines = [ln for ln in lines2 if ln.startswith("[runoff]")]
    assert runoff_lines, f"runoff never announced:\n{chr(10).join(lines2)}"
    if any("winner" in ln for ln in runoff_lines):
        # A real runoff ran: the extra batch (index generations+1) was drawn.
        assert (2 + 1) in seen_rounds, \
            f"runoff must draw batch index {2 + 1}; saw {seen_rounds}"
    else:
        # Only one distinct leader -> skip is the correct behaviour.
        assert any("skipped" in ln for ln in runoff_lines), runoff_lines
    assert best2.evaluation is not None
    print(f"[B2h] patience+runoff OK: patience stopped after rounds {seen_rounds[:1]}"
          f"... (early), runoff lines: {len(runoff_lines)}, "
          f"winner {best2.name!r}")


_CRASHING_COMBINER_CODE = '''
_SEEN = [0]

def skill_scores(driver_obs, phi_ep, phi_step, w=None):
    # Survives the sandbox's synthetic probe (a handful of calls) and then raises
    # for the rest of the episode -- the failure mode the RUNTIME rule exists for:
    # validation-clean code that still dies on the ten-thousandth real driver.
    _SEEN[0] += 1
    if _SEEN[0] > 50:
        raise KeyError("idle_min")
    return {"eff": float(driver_obs.get("num_assigned", 0)), "fair": 0.0}
'''


class _CrashThenFixClient(ScriptedClient):
    """Serves a combiner that raises on every driver for the first ``n_bad``
    combiner requests, then the normal working one.

    ``n_bad=1`` exercises "one repair fixes it"; a huge ``n_bad`` exercises "the
    repair did not take -> ELIMINATED"."""

    def __init__(self, n_bad: int = 1):
        super().__init__()
        self.n_bad = n_bad
        self.combiner_calls = 0

    def complete(self, system: str, user: str, *, temperature=None) -> str:
        if "skill_scores" in user:
            self.combiner_calls += 1
            if self.combiner_calls <= self.n_bad:
                self.calls += 1
                return json.dumps({
                    "combiner_name": "crashes_on_every_driver",
                    "strategy": "Read a missing obs key.",
                    "description": "Intentionally raises inside skill_scores so "
                                   "every weight decision falls back.",
                    "code": _CRASHING_COMBINER_CODE,
                })
        return super().complete(system, user, temperature=temperature)


def test_b2h_fallback_repairs_then_eliminates() -> None:
    """A runtime fallback must trigger ONE targeted repair, and a program that
    still falls back afterwards must be eliminated -- not quietly ranked.

    v5 only subtracted ``fallback_penalty * fallback_rate``, so a program that
    crashed on 3% of its drivers and coasted on the default skill could still win
    a round. v6 treats any fallback as a bug: the repair prompt carries the real
    cause (``KeyError: 'idle_min'`` here), and the fix replaces the program in the
    round."""
    from pref_dispatch.llm.evolve_combiner import evolve_combiner_objectives

    skills, cards = _two_skill_basis()
    scenarios = _small_scenarios(2, base_seed=0)
    objectives = ObjectiveSampler(rng=random.Random(6)).sample_batch(2)

    # (a) one bad proposal -> the repair takes; the winner must be the FIX.
    lines: list = []
    best = evolve_combiner_objectives(
        _CrashThenFixClient(n_bad=1), _STUB_PROFILE, skills, cards,
        scenarios, objectives, generations=0, mu=1, lam=1,
        rng=random.Random(0), log=lines.append,
    )
    joined = "\n".join(lines)
    assert "[runtime]" in joined, f"fallback was not detected:\n{joined}"
    assert "KeyError" in joined, f"repair prompt never saw the real cause:\n{joined}"
    assert best.combiner.fallback_rate == 0.0, \
        f"winner still falls back: {best.combiner.fallback_rate}"
    assert best.meta.get("operator", "").startswith("runtime-repair"), \
        f"winner is not the repaired program: {best.meta.get('operator')!r}"
    assert "ELIMINATED" not in joined, f"a repaired program was eliminated:\n{joined}"

    # (b) every proposal crashes -> the repair does not take -> ELIMINATED. The
    # run still returns something (an empty archive would kill the run), but the
    # log must say so rather than treat it as a normal contender.
    lines2: list = []
    best2 = evolve_combiner_objectives(
        _CrashThenFixClient(n_bad=99), _STUB_PROFILE, skills, cards,
        scenarios, objectives, generations=0, mu=1, lam=1,
        rng=random.Random(0), log=lines2.append,
    )
    joined2 = "\n".join(lines2)
    assert "ELIMINATED" in joined2, \
        f"a permanently-crashing program was not eliminated:\n{joined2}"
    assert best2.combiner.fallback_rate > 0.5, best2.combiner.fallback_rate
    print(f"[B2h] fallback rule OK: one repair fixed the crasher -> "
          f"{best.name!r} (fallback 0.00, operator {best.meta['operator']}); "
          f"unfixable crasher eliminated (fallback "
          f"{best2.combiner.fallback_rate:.2f})")


def test_b2h_yardstick_out_of_group() -> None:
    """The frozen single skills must be OUT of the selection group by default and
    still reachable as the fixed-batch yardstick.

    ``skill_refs_in_group=False`` is the v6 default; the yardstick is the only
    place the 8 skill rollouts are paid for, and it is what the paper's "beats N
    of 8 single skills" number comes from."""
    import math

    from pref_dispatch.llm.evolve_combiner import (
        evolve_combiner_objectives,
        skill_yardstick,
    )

    skills, cards = _two_skill_basis()
    scenarios = _small_scenarios(2, base_seed=0)
    objectives = ObjectiveSampler(rng=random.Random(4)).sample_batch(2)

    best = evolve_combiner_objectives(
        ScriptedClient(), _STUB_PROFILE, skills, cards, scenarios, objectives,
        generations=0, mu=2, lam=1, rng=random.Random(0), log=lambda *_: None,
    )
    card = skill_yardstick(best, skills, scenarios, objectives, log=lambda *_: None)
    assert set(card) == {"rank", "beaten", "n_skills", "per_family",
                         "skill_rewards", "champion_rewards"}, sorted(card)
    assert card["n_skills"] == len(skills)
    from pref_dispatch.llm.group_fitness import Z_CLIP
    assert -Z_CLIP <= card["rank"] <= Z_CLIP, card["rank"]
    assert set(card["beaten"]) <= set(skills), card["beaten"]
    assert math.isfinite(card["champion_rewards"])
    print(f"[B2h] yardstick OK: advantage {card['rank']:+.2f} vs {card['n_skills']} "
          f"frozen skills, beats {len(card['beaten'])} "
          f"({', '.join(card['beaten']) or 'none'})")


def test_b2i_llm_proposed_briefs() -> None:
    """v6 item 7: the natural-language objectives must be WRITTEN by the model,
    fresh per batch, and must never be able to stall training.

    Four things are checked, all key-free with a scripted client:
      1. a good reply yields n briefs, none of them one already used;
      2. junk / code / duplicate / one-word entries are dropped, not accepted;
      3. a first junk reply is retried with feedback and the second reply lands;
      4. an always-broken client returns nothing and the sampler KEEPS its
         current pool (a bad proposal round costs one call, not the run).
    """
    from pref_dispatch.llm.objective_sampler import (
        DEFAULT_NL_BRIEFS,
        ObjectiveSampler,
        propose_briefs,
    )

    _GOOD = [
        "pay only for drop-offs completed during the busiest hour and nothing else",
        "refuse to pay for any minute a car spends empty or standing still",
        "treat one ride carrying four parties as worth more than six solo trips",
        "keep pickup distance short even when that means turning down long fares",
    ]

    class _BriefClient:
        """Replies to the brief-proposal prompt only; counts calls."""

        def __init__(self, replies):
            self.replies = list(replies)
            self.calls = 0
            self.seen_used = []

        def complete(self, system, user, *, temperature=None):
            assert "objective" in system.lower(), system
            self.seen_used.append(user)
            self.calls += 1
            i = min(self.calls - 1, len(self.replies) - 1)
            return self.replies[i]

    # 1. Clean reply.
    c1 = _BriefClient([json.dumps({"briefs": _GOOD, "why_different": "..."})])
    got = propose_briefs(c1, 4, DEFAULT_NL_BRIEFS, log=lambda *_a: None)
    assert got == _GOOD, got
    assert c1.calls == 1
    # The prompt must actually list what has been used, or the model cannot avoid it.
    assert DEFAULT_NL_BRIEFS[0] in c1.seen_used[0]

    # 2. Junk entries are filtered; only the one real brief survives.
    dirty = json.dumps({"briefs": [
        DEFAULT_NL_BRIEFS[0].upper(),                       # a repeat (case/space-blind)
        "def reward(event): return event['revenue']",       # code, not a brief
        "revenue = 2 * completions",                        # a formula
        "more",                                             # too short
        _GOOD[1],                                           # the one keeper
        _GOOD[1] + "  ",                                    # dup of the keeper
    ]})
    c2 = _BriefClient([dirty])
    got2 = propose_briefs(c2, 6, DEFAULT_NL_BRIEFS, log=lambda *_a: None)
    assert got2 == [_GOOD[1]], got2

    # 3. Junk first, good second -> one retry, then success.
    c3 = _BriefClient(["not json at all",
                       json.dumps({"briefs": _GOOD[:2]})])
    got3 = propose_briefs(c3, 2, (), log=lambda *_a: None)
    assert got3 == _GOOD[:2], got3
    assert c3.calls == 2, c3.calls

    # 4. Always broken -> [] , and the sampler keeps the bank it already had.
    c4 = _BriefClient(["<html>gateway timeout</html>"])
    assert propose_briefs(c4, 3, (), attempts=2, log=lambda *_a: None) == []
    sam = ObjectiveSampler(client=c4, rng=random.Random(0), log=lambda *_a: None)
    before = sam.nl_briefs
    sam.refresh_briefs()
    assert sam.nl_briefs == before, "a failed proposal round must not empty the pool"

    # 5. A working client swaps the pool, and used briefs accumulate so the next
    #    round is told to move past them.
    c5 = _BriefClient([json.dumps({"briefs": _GOOD})])
    sam5 = ObjectiveSampler(client=c5, rng=random.Random(0), log=lambda *_a: None)
    assert sam5.llm_briefs is True
    assert list(sam5._used_briefs) == list(DEFAULT_NL_BRIEFS)
    sam5.refresh_briefs()
    assert sam5.nl_briefs == tuple(_GOOD), sam5.nl_briefs
    sam5._remember_briefs([
        _mk_nl_stub(_GOOD[0]), _mk_nl_stub(_GOOD[0]), _mk_nl_stub(_GOOD[2]),
    ])
    assert sam5._used_briefs[-2:] == [_GOOD[0], _GOOD[2]], sam5._used_briefs[-2:]

    # 6. No client -> no proposal path at all (the key-free run is unchanged).
    sam6 = ObjectiveSampler(rng=random.Random(0), log=lambda *_a: None)
    assert sam6.llm_briefs is False
    sam6.refresh_briefs()
    assert sam6.nl_briefs == tuple(DEFAULT_NL_BRIEFS)

    print(f"[B2i] LLM-proposed briefs OK: 4 fresh briefs accepted, "
          f"{6 - 1} junk/dup entries filtered, 1 retry recovered a bad reply, "
          f"broken client left the {len(DEFAULT_NL_BRIEFS)}-brief bank intact")


def _mk_nl_stub(brief: str):
    """A minimal SampledObjective carrying just a brief (for _remember_briefs)."""
    from pref_dispatch.llm.objective_sampler import SampledObjective

    return SampledObjective(label="nl", family="nl", reward_function=None, w=None,
                            meta={"brief": brief})


def _mk_candidate(name: str, code: str, skill_names):
    """A CombinerCandidate straight from source (no LLM), for the parallel test."""
    from pref_dispatch.llm.evolve_combiner import CombinerCandidate
    from pref_dispatch.llm.sandbox import compile_combiner

    return CombinerCandidate(
        meta={"combiner_name": name, "strategy": "offline test",
              "description": "offline test", "code": code, "gen": 0},
        scorer=compile_combiner(code), skill_names=list(skill_names),
    )


def test_b2j_parallel_matches_sequential() -> None:
    """Spreading the rollouts over processes must not move a single number.

    This is the whole correctness claim of :mod:`pref_dispatch.llm.parallel`: a
    rollout is seeded by its own scenario, so WHICH process ran it and in what
    order cannot change its reward. The check rolls the same two combiners and
    the same two single skills both ways on the same tiny batch and demands the
    reward rows be EQUAL (not close -- equal), plus that a candidate carrying no
    source is refused up front rather than half-running the round.
    """
    import math
    import pickle

    from pref_dispatch.llm.combiner_eval import (
        _roll_pair_rewards,
        _skill_reference_rewards,
        blindness_from_dists,
    )
    from pref_dispatch.llm.parallel import (
        NotParallelizable,
        parallel_pair_rewards,
        parallel_skill_rows,
        resolve_workers,
    )

    skills, _ = _two_skill_basis()
    scenarios = _small_scenarios(2, base_seed=0)
    objectives = ObjectiveSampler(rng=random.Random(11)).sample_batch(2)
    cands = [_mk_candidate("balanced", _COMBINER_CODE, skills),
             _mk_candidate("flipper", _DIFF_COMBINER_CODE, skills)]

    seq_rows = [_roll_pair_rewards(c.make_combiner(), skills, scenarios, objectives)
                for c in cands]
    recs = parallel_pair_rewards(cands, skills, scenarios, objectives,
                                 workers=2, log=lambda *_: None)
    assert len(recs) == len(cands), len(recs)
    for c, seq, rec in zip(cands, seq_rows, recs):
        assert rec["rewards"] == seq, (c.name, rec["rewards"], seq)
        assert rec["n_calls"] > 0, (c.name, rec["n_calls"])
        assert rec["n_fallbacks"] == 0, (c.name, rec["reason"])
        # Blindness travels back as fleet mixes, not as 400 observation triples.
        b = blindness_from_dists(rec["picks"] or [])
        assert math.isfinite(b) and 0.0 <= b <= 1.0, b

    seq_refs = _skill_reference_rewards(skills, scenarios, objectives)
    par_refs = parallel_skill_rows(skills, scenarios, objectives,
                                   workers=2, log=lambda *_: None)
    assert par_refs == seq_refs, (par_refs, seq_refs)

    # A candidate with no source cannot be described for a worker: the whole call
    # must refuse, so the caller drops to the in-process loop with nothing wasted.
    naked = _mk_candidate("naked", _COMBINER_CODE, skills)
    naked.meta.pop("code")
    try:
        parallel_pair_rewards([naked], skills, scenarios, objectives,
                              workers=2, log=lambda *_: None)
    except NotParallelizable:
        pass
    else:
        raise AssertionError("a source-less candidate must raise NotParallelizable")

    assert resolve_workers(1) == 1 and resolve_workers(0) == 1
    assert resolve_workers(None) >= 1

    # A FROZEN evolved skill is a module built at runtime from a .py on disk: the
    # object cannot be pickled and its own imports mean the restricted exec cannot
    # take its source either, so the PATH must travel. This branch is what the real
    # 8-skill basis uses -- the offline basis above is all sandbox-compiled source.
    import os
    import tempfile

    from pref_dispatch.llm.basis import _load_evolved_module
    from pref_dispatch.llm.parallel import _worker_skills, skills_payload

    with tempfile.TemporaryDirectory() as td:
        py = os.path.join(td, "frozen_probe.py")
        with open(py, "w", encoding="utf-8") as f:
            f.write("import math\n\n\n"
                    "def score(driver_obs, order, phi_ep, phi_step):\n"
                    "    return math.sqrt(4.0)\n")
        frozen = _load_evolved_module(py, "frozen_probe")
        try:
            pickle.dumps(frozen)
        except Exception:
            pass
        else:
            raise AssertionError("a frozen evolved skill should not pickle; if it "
                                 "now does, the 'path' branch is dead code")
        pay = skills_payload({"frozen_probe": frozen})
        assert pay[0]["kind"] == "path", pay
        assert len(pickle.dumps(pay)) < 4096, "payload should be a path, not a blob"
        rebuilt = _worker_skills(pay)
        assert rebuilt["frozen_probe"].score(None, None, None, None) == 2.0

    print(f"[B2j] parallel == sequential OK: {len(cands)} combiners x "
          f"{len(scenarios)} pairs and {len(skills)} single-skill rows identical "
          f"on 2 processes; source-less candidate refused up front; frozen "
          f"evolved skill ships as a path")


def test_b2f_stratified_scenes() -> None:
    """The stratified scene batch must ACTUALLY stratify: every (fleet-band x
    regime) cell appears, the fleet draw stays INSIDE its cell's band (the
    override must reach the fleet sampler -- a silent no-op here is what v2's
    first retrain log caught: fleet 1815 inside a (500,1000) cell), order caps
    alternate capped/full-hour, and seeds stay pinned to base_seed + i."""
    from pref_dispatch.scenario import ScenarioRanges, ScenarioSampler

    bands = [(200, 500), (500, 1000), (1000, 1500)]
    regimes = ["peak", "offpeak"]
    order_modes = [600, 800, 1500]
    base = ScenarioRanges(fleet=(400, 2500), fleet_dist="loguniform",
                         order_limit=None, order_limits=None)
    sam = ScenarioSampler(ranges=base, rng=random.Random(0), split="train")
    batch = sam.sample_batch_stratified(12, bands=bands, regimes=regimes,
                                        order_modes=order_modes, base_seed=7)

    assert len(batch) == 12
    cells_seen = {(lo, hi, reg): 0 for (lo, hi) in bands for reg in regimes}
    for i, sc in enumerate(batch):
        band = bands[(i // 2) % len(bands)]
        reg = regimes[i % 2]
        cell = (band[0], band[1], reg)
        cells_seen[cell] += 1
        # THE regression: fleet must land inside its cell's band, not the base range.
        assert band[0] <= sc.num_drivers <= band[1], (
            f"scenario {i}: fleet {sc.num_drivers} outside cell band "
            f"({band[0]},{band[1]}) -- the ranges override is not reaching the fleet sampler"
        )
        assert sc.regime == reg, f"scenario {i}: regime {sc.regime} != {reg}"
        if i % 2 == 0:
            assert sc.order_limit == order_modes[i % len(order_modes)], (
                f"scenario {i}: capped cell must use its order mode, got {sc.order_limit}")
        else:
            assert sc.order_limit is None, f"scenario {i}: odd cell must be full hour"
        assert sc.seed == 7 + i, f"scenario {i}: seed {sc.seed} != 7 + {i}"
    assert all(v == 2 for v in cells_seen.values()), cells_seen
    print(f"[B2f] stratified scenes OK: all {len(batch)} scenarios in-band "
          f"(fleets {[s.num_drivers for s in batch]}), regimes/caps alternate, "
          f"seeds pinned 7..18")


# --------------------------------------------------------------------------- #
# B2g: the sampling fixes for the 2026-08-09 gate failure (19/30, completion
#      1/6): every objective family must be seen at BOTH a scarce and a large
#      fleet -- family counts >= 2 and paired across fleet bands.
# --------------------------------------------------------------------------- #
def test_b2g_structural_round_robin() -> None:
    """No structural sub-family may come out 0 (v2 drew pooling 0, completion 1)."""
    from collections import Counter

    s = ObjectiveSampler(rng=random.Random(0), structural_fraction=0.5)
    counts = Counter(o.family for o in s.sample_batch(12))
    for fam in ("completion", "pooling"):
        assert counts[fam] >= 2, (
            f"{fam} drawn {counts[fam]}x in a 12-objective batch -- with < 2 draws "
            f"it cannot be seen at both a scarce and a large fleet: {dict(counts)}")
    # The legacy default share must be unchanged for every other caller.
    legacy = ObjectiveSampler(rng=random.Random(0))
    assert legacy.structural_fraction == ObjectiveSampler.MIN_STRUCTURAL_FRACTION
    print(f"[B2g] structural round-robin OK: {dict(counts)} (each of "
          f"completion/pooling >= 2 at structural_fraction=0.5)")


def test_b2g_family_band_pairing() -> None:
    """Each family's draws must land in DIFFERENT fleet bands after pairing.

    This is the fix for the gate's completion failure: v2 trained completion on a
    single fleet-938 scene, so the champion learned the at-scale rule ("chase
    fast, low-detour trips") and applied it at fleet 200-500, where refusing
    orders costs 13-19.5% completion and reading w was worth -267..-593 reward.

    Checked over MANY seeds, because the property has to come from the ALGORITHM.
    The first version of this test asserted only ``0 in seen["completion"]`` on one
    seed and passed -- while the run it was meant to protect put completion on
    fleet 938 and 1137, both mid/large. The band a family got was decided by its
    position in an alphabetical sort, i.e. by luck.
    """
    from pref_dispatch.llm.run_phase2_full import pair_by_fleet_band
    from pref_dispatch.scenario import ScenarioRanges, ScenarioSampler

    bands = [(200, 500), (500, 1000), (1000, 1500)]
    scarce, largest = 0, len(bands) - 1
    reports = []
    for seed in range(8):
        sam = ScenarioSampler(
            ranges=ScenarioRanges(fleet=(400, 2500), fleet_dist="loguniform"),
            rng=random.Random(seed), split="train")
        scenarios = sam.sample_batch_stratified(
            12, bands=bands, regimes=["peak", "offpeak"],
            order_modes=[600, 800, 1500], base_seed=seed)
        objectives = ObjectiveSampler(
            rng=random.Random(seed + 3), structural_fraction=0.5).sample_batch(12)

        paired = pair_by_fleet_band(scenarios, objectives, bands)
        assert len(paired) == 12 and all(o is not None for o in paired)
        assert sorted(id(o) for o in paired) == sorted(id(o) for o in objectives), \
            f"seed {seed}: pairing must PERMUTE the same objectives, not drop/dup"

        def _band(sc):
            return next(bi for bi, (lo, hi) in enumerate(bands)
                        if lo <= sc.num_drivers <= hi)

        seen: Dict[str, set] = {}
        for sc, ob in zip(scenarios, paired):
            seen.setdefault(ob.family, set()).add(_band(sc))
        counts = {f: sum(1 for o in paired if o.family == f) for f in seen}

        # Capacity bound, NOT "every family spreads": a 12-scene batch has only
        # 4 slots per band, so if 6 families draw >= 2 each they cannot all get a
        # scarce slot. What must hold is that the spread goes as far as the slots
        # allow -- so at least as many families span 2+ bands as the scarce band
        # has room for. (Catches a regression that stops spreading altogether.)
        multi = [f for f in seen if counts[f] >= 2]
        spread = [f for f in multi if len(seen[f]) >= 2]
        assert len(spread) >= min(len(multi), 4), (
            f"seed {seed}: only {len(spread)} of {len(multi)} multi-draw families "
            f"span 2+ bands; scarce band has 4 slots so >= {min(len(multi), 4)} must")
        # The contract that decides the gate: each of the STRUCTURAL families is
        # trained at BOTH scarcity and scale. These get first claim precisely
        # because a scale-specific rule learned from one band is what cost v2 the
        # scarce-fleet completion cells; a linear-coefficient draw carries no such
        # rule to misgeneralise.
        for fam in ("completion", "pooling"):
            assert counts.get(fam, 0) >= 2, f"seed {seed}: {fam} drew {counts.get(fam, 0)}"
            assert scarce in seen[fam], (
                f"seed {seed}: {fam} never sees the SCARCE band {bands[scarce]} "
                f"(bands {sorted(seen[fam])}) -- this is the v2/v3 gate failure")
            assert largest in seen[fam], (
                f"seed {seed}: {fam} never sees the LARGEST band {bands[largest]} "
                f"(bands {sorted(seen[fam])}) -- it would lose the at-scale cells")
        reports.append({f: sorted(b) for f, b in sorted(seen.items())})

    print(f"[B2g] family x band pairing OK over {len(reports)} seeds: "
          f"completion/pooling each span the scarce AND largest band "
          f"on every seed. seed0 {reports[0]}")


def test_b2g_extract_near_json() -> None:
    """Near-JSON must parse: it killed generation 0 of three Phase-2 runs.

    All three died on ``Expecting property name enclosed in double quotes: line 1
    column 2`` -- the payload's very first key was single-quoted. Generation 0 is
    the one call whose failure ends the whole run (later generations log and carry
    on), so the parser gets a third tier and the repair pass retries COOLER."""
    from pref_dispatch.llm.extract import ExtractionError, extract_json

    # The failing shape: single-quoted keys, trailing comma, True, and raw
    # newlines inside the code payload (invalid as a python literal, so the
    # relaxed JSON rewrite -- not literal_eval -- is what recovers it).
    payload = (
        "{'name': 'c1', 'description': 'd', "
        "'code': 'def combine(x):\n    return x\n', 'ok': True,}"
    )
    obj = extract_json(payload)
    assert sorted(obj) == ["code", "description", "name", "ok"], obj
    assert obj["ok"] is True and obj["name"] == "c1"
    compile(obj["code"], "<candidate>", "exec")      # payload survives intact

    # BARE unquoted keys: the JS-literal drift that actually produces the
    # char-1 failure (the single-quote theory did not survive contact -- the
    # relaxed rewrite handled quotes and the payload still failed at char 1).
    obj = extract_json('{combiner_name: "c", strategy: "s", code: "def f():\n    pass\n"}')
    assert sorted(obj) == ["code", "combiner_name", "strategy"], obj
    assert extract_json('{a: 1, "b": {c: [1, 2], d: true}, e: None,}') == \
        {"a": 1, "b": {"c": [1, 2], "d": True}, "e": None}
    # A colon inside a string is not a key boundary.
    assert extract_json('{note: "time: 10:30", flag: True}') == \
        {"note": "time: 10:30", "flag": True}
    # A bare VALUE is not a key: this must stay unparseable, not become a string.
    try:
        extract_json("{a: bogus}")
        raise AssertionError("a bare value must not be coerced into validity")
    except ExtractionError:
        pass

    # Fences + trailing commas, and valid JSON must still take tier 1 unchanged.
    assert extract_json('pre\n```json\n{"a": 1, "b": [1,2,],}\n```') == {"a": 1, "b": [1, 2]}
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json("<think>reasoning</think>\n{'a': 2}") == {"a": 2}
    # A bare " inside a single-quoted string must not break the rewrite.
    assert extract_json("{'d': 'he said \"hi\"'}") == {"d": 'he said "hi"'}

    # Garbage must still fail loudly, and no tier may EXECUTE anything.
    for hostile in ("no object here", "{'a': __import__('os').getcwd()}",
                    "{'a': open('x')}"):
        try:
            extract_json(hostile)
            raise AssertionError(f"must not parse/execute: {hostile}")
        except ExtractionError:
            pass
    print("[B2g] near-JSON extraction OK: single-quoted keys + raw newlines in "
          "code recovered, valid JSON unchanged, calls refused not executed")


def test_b2g_repair_cools_down() -> None:
    """A repair retry must ask COOLER, not re-roll the same temperature.

    v1 succeeded at temperature 0.9; v2 and v3 both failed at 1.0. Malformed
    output is the failure mode high temperature causes, so recovery lowers it."""
    from pref_dispatch.llm import evolve_combiner as ec

    seen: List[Optional[float]] = []

    good = json.dumps({
        "combiner_name": "cooled_combiner",
        "strategy": "flat blend over the frozen basis",
        "description": "Scores every frozen skill equally; used to test the "
                       "repair pass, not to dispatch well.",
        "code": "def skill_scores(driver_obs, phi_ep, phi_step, w=None):\n"
                "    return {'a': 1.0}\n",
    })

    class _Client:
        def complete(self, system, user, temperature=None):
            seen.append(temperature)
            # Fail the first two attempts the way the real runs failed: a
            # single-quoted payload that json.loads rejects... except the tier-3
            # rewrite now recovers that, so fail with something truly unparseable.
            if len(seen) <= 2:
                return "I cannot comply with that request."
            return good

    # Divert the unparseable-payload dump so this test cannot litter the real
    # diagnostic directory with its own synthetic failures. The dump directory lives
    # in the SHARED repair module (all three phases use it), so that is what has to
    # be patched -- rebinding a re-exported alias in evolve_combiner would silently
    # do nothing.
    import tempfile

    from pref_dispatch.llm import repair as _repair

    real_dir = _repair.UNPARSEABLE_DIR
    with tempfile.TemporaryDirectory() as tmp:
        _repair.UNPARSEABLE_DIR = tmp
        try:
            cand = ec._propose_with_repair(
                _Client(), lambda fb: {"system": "s", "user": "u"}, gen=0,
                skill_names=["a"], temperature=1.0, log=lambda m: None)
            dumped = sorted(os.listdir(tmp))
        finally:
            _repair.UNPARSEABLE_DIR = real_dir
    assert len(dumped) == 2, f"both failed completions must be dumped: {dumped}"
    assert cand is not None
    assert seen == [1.0, 0.75, 0.5], f"expected a cooling schedule, got {seen}"
    assert seen[0] == 1.0, "the FIRST attempt must keep the requested diversity"
    assert all(t >= ec._REPAIR_MIN_TEMPERATURE for t in seen), seen
    print(f"[B2g] repair cool-down OK: temps {seen} (first attempt keeps 1.0, "
          f"retries cool by {ec._REPAIR_COOLDOWN})")


def test_b2g_freeze_never_clobbers() -> None:
    """Freezing must never overwrite an existing artifact.

    The MODEL picks the combiner name and reuses good ones: a v4 run proposed
    ``objective_shape_dispatcher_v3``, the exact name of the frozen v2 champion
    still needed for comparison. ``evolved/`` is untracked, so an overwrite is
    unrecoverable. The recorded ``combiner_name`` must follow the file name so
    ``--combiner-name`` still resolves it."""
    import tempfile

    from pref_dispatch.llm.evolve_combiner import _unique_frozen_name

    with tempfile.TemporaryDirectory() as d:
        assert _unique_frozen_name("foo", d) == "foo"      # free name kept as-is
        open(os.path.join(d, "foo.py"), "w").write("x")
        assert _unique_frozen_name("foo", d) == "foo_r2"   # collision -> suffix
        open(os.path.join(d, "foo_r2.py"), "w").write("x")
        assert _unique_frozen_name("foo", d) == "foo_r3"   # walks past taken ones
        assert _unique_frozen_name("bar", d) == "bar"      # unrelated name free
    print("[B2g] freeze collision-safe: name -> name_r2 -> name_r3, existing "
          "artifacts never overwritten")


def main() -> None:
    test_objective_sampler()
    test_phase1_directed()
    test_b1_replace_loop()
    test_b1_fill_no_gate()
    test_b1_qd_uses_group_search()
    test_b1_directed_uses_group_search()
    test_b1_banded_windows()
    test_leader_checkpoint()
    test_b1_discard_evicted()
    test_b2_combiner_objectives()
    test_b2_blindness_discriminates()
    test_b2_wrong_flip_loses_in_group()
    test_b2e_group_relative_scale_invariance()
    test_b2e_group_relative_discriminates()
    test_b2e_skill_anchor()
    test_b2e_crash_is_the_baseline()
    test_b2e_delta_sign_is_absolute()
    test_b2_stratified_batch()
    test_b2f_selection_score_bias()
    test_b2f_accept_rule()
    test_b2f_stratified_scenes()
    test_b2h_family_elite_slot()
    test_b2h_mu_lambda_rotating_scenes()
    test_b2h_patience_and_runoff()
    test_b2h_fallback_repairs_then_eliminates()
    test_b2h_yardstick_out_of_group()
    test_b2i_llm_proposed_briefs()
    test_b2j_parallel_matches_sequential()
    test_b2g_structural_round_robin()
    test_b2g_family_band_pairing()
    test_b2g_extract_near_json()
    test_b2g_repair_cools_down()
    test_b2g_freeze_never_clobbers()
    test_b3_repositioner_objectives()
    print("\nALL Part-B offline checks passed (no API key used).")


if __name__ == "__main__":
    main()
