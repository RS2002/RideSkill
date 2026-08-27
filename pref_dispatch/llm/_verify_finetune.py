"""Offline verification for Phase-3 warm-start fine-tune (no LLM key needed).

Checks:
 1. seed-unit: evolve_combiner(seed_code=..., generations=0) returns the frozen
    combiner as the incumbent WITHOUT any LLM call (client that raises if touched),
    and its fitness == directly evaluating the base combiner on the scenario.
 2. skill-select: the captured fleet-pick set is a subset of the basis and its
    fractions sum to ~1.
 3. scenario_tag is filesystem-safe.
 4. load_basis(extra_skill_dirs=...) overlays + overrides a same-name skill.
 5. partial-spec: a ScenarioSpec with only some axes pinned domain-randomizes the
    rest across the batch (pinned axes fixed, unspecified axes vary), and the
    reward-contract reconstruction picks env_reward for the default-reward combiner
    but scalarize for the preference combiner.
"""
from __future__ import annotations

import json
import os
import tempfile

from pref_dispatch.llm.basis import (
    EvolvedSkillsDir,
    load_basis,
    load_frozen_combiner,
)
from pref_dispatch.llm.combiner_eval import (
    evaluate_combiner_scenarios,
    scenario_norm_frames,
)
from pref_dispatch.llm.combiner_adapter import NO_PICK
from pref_dispatch.llm.evolve_combiner import evolve_combiner
from pref_dispatch.llm.finetune import (
    ScenarioSpec,
    _reward_from_combiner_meta,
    _select_skills,
    scenario_tag,
    spec_tag,
)
from pref_dispatch.scenario import Scenario


class _ExplodingClient:
    """Any LLM call must fail this test -- warm-start gen-0 must not call the model."""

    def complete(self, *a, **k):
        raise AssertionError("LLM was called during a warm-start seed / gens=0 run!")


def _small_scenario() -> Scenario:
    # Tiny fleet + capped orders so the offline rollouts are fast.
    return Scenario(
        num_drivers=120, driver_capacity=4, speed_kmh=35.0, regime="offpeak",
        split="train", order_limit=150, pref_revenue=0.5, seed=0,
    )


def test_seed_unit_no_llm() -> None:
    skills, cards = load_basis(include_evolved=True)
    # ``None`` = whichever combiner is frozen on disk. These checks are about the
    # fine-tune plumbing, not about which champion is current.
    combiner, meta = load_frozen_combiner(None, skill_names=tuple(skills))
    sc = _small_scenario()
    frames = scenario_norm_frames(skills, [sc])

    # gens=0 warm-start: must NOT call the (exploding) client.
    best = evolve_combiner(
        _ExplodingClient(), "profile", skills, cards, [], None,
        scenarios=[sc], scenario_frames=frames,
        objective="scalarize",
        seed_code=meta["code"], seed_meta=meta,
        generations=0, lam=1, seed=sc.seed, log=lambda *_: None,
    )
    # Incumbent fitness == directly evaluating the base combiner on this scenario.
    direct = evaluate_combiner_scenarios(
        combiner, skills, [sc], frames, objective="scalarize",
    )
    assert abs(best.evaluation.fitness - direct.fitness) < 1e-9, (
        best.evaluation.fitness, direct.fitness
    )
    assert "def skill_scores" in best.meta["code"]
    print(f"[1] seed-unit OK: warm-start gens=0 fitness={best.evaluation.fitness:.4g} "
          f"== direct base eval (no LLM call).")


def test_skill_select() -> None:
    skills, _cards = load_basis(include_evolved=True)
    combiner, _meta = load_frozen_combiner(None, skill_names=tuple(skills))
    sc = _small_scenario()
    picked = _select_skills(combiner, skills, sc, threshold=0.01, log=lambda *_: None)
    # NO_PICK is the "assign nothing this step" sentinel, not a skill: the
    # combiner may legitimately return it alongside real skills, so it is
    # excluded before the subset check rather than treated as a stray name.
    assert set(picked) - {NO_PICK} <= set(skills), (picked, list(skills))
    fracs = combiner.fleet_pick_fractions(sc.preference)
    assert abs(sum(fracs.values()) - 1.0) < 1e-6, sum(fracs.values())
    print(f"[2] skill-select OK: picked {picked} subset of basis; "
          f"fractions sum={sum(fracs.values()):.4f}.")


def test_tag_safe() -> None:
    sc = _small_scenario()
    tag = scenario_tag(sc)
    assert "/" not in tag and "\\" not in tag and " " not in tag, tag
    assert tag == os.path.basename(tag)
    print(f"[3] tag OK: {tag!r}")


def test_extra_skill_dirs_override() -> None:
    # Copy an evolved skill into a temp dir but overwrite its objective marker in the
    # card, and confirm load_basis(extra_skill_dirs=...) uses the OVERRIDE version.
    import glob
    import json
    import shutil

    metas = glob.glob(os.path.join(EvolvedSkillsDir, "*.meta.json"))
    assert metas, "need at least one evolved skill to test override"
    name = os.path.basename(metas[0])[: -len(".meta.json")]
    with tempfile.TemporaryDirectory() as td:
        shutil.copy(os.path.join(EvolvedSkillsDir, f"{name}.py"),
                    os.path.join(td, f"{name}.py"))
        m = json.load(open(metas[0], encoding="utf-8"))
        m["objective"] = "OVERRIDE_MARKER"
        json.dump(m, open(os.path.join(td, f"{name}.meta.json"), "w", encoding="utf-8"))

        base_skills, base_cards = load_basis(include_evolved=True)
        ov_skills, ov_cards = load_basis(include_evolved=True, extra_skill_dirs=[td])
        assert set(ov_skills) == set(base_skills), "override must not add/drop skills"
        card = next(c for c in ov_cards if c["skill_name"] == name)
        assert card["objective"] == "OVERRIDE_MARKER", card
    print(f"[4] extra_skill_dirs OK: {name!r} objective overridden, basis set unchanged.")


def test_partial_spec_randomizes() -> None:
    # Only fleet pinned -> capacity/speed/regime/preference must be RANDOMIZED, and
    # they must actually vary across the batch (not silently collapse to a default).
    spec = ScenarioSpec(num_drivers=1000, n_scenarios=4, split="test", seed=7)
    assert not spec.is_concrete(), spec
    assert set(spec.randomized) == {
        "driver_capacity", "speed_kmh", "regime", "pref_revenue"
    }, spec.randomized
    batch = spec.build_scenarios()
    assert len(batch) == 4, len(batch)
    for sc in batch:
        assert sc.num_drivers == 1000, ("pinned fleet must hold", sc.num_drivers)
    # At least one randomized axis genuinely varies across the batch.
    caps = {sc.driver_capacity for sc in batch}
    regimes = {sc.regime for sc in batch}
    assert len(caps) > 1 or len(regimes) > 1, (caps, regimes)
    tag = spec_tag(spec, batch[0])
    assert tag.startswith("ft_f1000_") and "rand4x4" in tag, tag
    assert "/" not in tag and "\\" not in tag and " " not in tag, tag

    # Fully-concrete spec reduces to the legacy single scenario + legacy tag.
    conc = ScenarioSpec(num_drivers=1000, driver_capacity=4, speed_kmh=35.0,
                        regime="peak", pref_revenue=0.5, seed=0, n_scenarios=1)
    assert conc.is_concrete(), conc
    only = conc.build_scenarios()
    assert len(only) == 1 and only[0].num_drivers == 1000
    assert spec_tag(conc, only[0]) == scenario_tag(only[0])
    print(f"[5a] partial-spec OK: fleet pinned, {spec.randomized} randomized over "
          f"{len(batch)} scenarios (caps={sorted(caps)}); concrete spec == legacy.")


def test_reward_contract_reconstruction() -> None:
    # The two no-compile branches of _reward_from_combiner_meta, driven by meta
    # dicts written here rather than by named artifacts on disk -- the contract
    # under test is the flag -> objective mapping, and pinning it to particular
    # champion filenames only made this file break whenever a version froze a
    # differently-named combiner (or dropped an old one).
    #
    # (a) frozen for the env's OWN reward (ignore_pref, no authored code) ->
    #     objective=env_reward with reward_function=None, NOT the old scalarize
    #     fallback, which would optimise a different objective.
    env_meta = {"ignore_pref": True,
                "reward_provenance": {"authored": False,
                                      "reward_name": "DefaultRewardFunction"}}
    rfn, rname, ig, _spec, obj = _reward_from_combiner_meta(env_meta)
    assert obj == "env_reward" and rfn is None and ig is True, (obj, rfn, ig)
    assert rname == "DefaultRewardFunction", rname
    # (b) a preference combiner with neither flag stays scalarize.
    pref_meta = {"objective": "scalarize"}
    rfn2, _n2, ig2, _s2, obj2 = _reward_from_combiner_meta(pref_meta)
    assert obj2 == "scalarize" and rfn2 is None and ig2 is False, (obj2, rfn2, ig2)

    # And the combiner actually frozen on disk must reconstruct CONSISTENTLY with
    # its own recorded flags -- this is the part that touches a real artifact,
    # without naming it.
    _c, live = load_frozen_combiner(None, skill_names=None)
    _r3, _n3, ig3, _s3, obj3 = _reward_from_combiner_meta(live)
    expect = "env_reward" if (live.get("ignore_pref")
                              or (live.get("reward_provenance") or {}).get("reward_name")
                              ) else "scalarize"
    assert obj3 == expect and ig3 is (expect == "env_reward"), (obj3, expect, ig3)
    print(f"[5b] reward-contract OK: ignore_pref+no-code -> env_reward "
          f"(reward_function=None); bare meta -> scalarize; frozen combiner "
          f"{live.get('name', '?')!r} -> {obj3}.")


if __name__ == "__main__":
    test_tag_safe()
    test_extra_skill_dirs_override()
    test_skill_select()
    test_partial_spec_randomizes()
    test_reward_contract_reconstruction()
    test_seed_unit_no_llm()
    print("\nALL PHASE-3 OFFLINE CHECKS PASSED")
