"""Concrete-behavior report: GIVEN a fixed platform reward, what strategy does the
combiner compose, and how does it actually behave?

Two report modes, chosen automatically from the frozen combiner's ``.meta.json``:

* **Single-objective (``ignore_pref=True``)** -- the §Phase-2 arm. The combiner was
  composed to maximise ONE fixed reward (LLM-authored from a preference, or the
  env's own reward_func) with NO runtime preference dial. The report reconstructs
  THAT reward, injects it into the anchor env, and prints ONE target reward, ONE
  fleet skill-mix distribution, ONE per-state pick column, and ONE KPI row under the
  authored reward. There is no preference sweep: one preference -> one reward -> one
  number.
* **Runtime-slider (``ignore_pref`` absent/False)** -- the legacy v2 arm. Prints the
  fleet skill-mix and per-driver picks across the revenue slider (rev 0.1/0.5/0.9)
  and a KPI row per preference point.

No LLM calls (the combiner is frozen pure Python). torch is NOT required.

    python -m pref_dispatch.llm.run_reward_strategy_report \
        --combiner reward_maximizing_dispatcher
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Tuple

from benchmark.config import make_benchmark_env
from benchmark.evaluate import TestSet, evaluate
from benchmark.run_generalize import anchor_cfg
from pref_dispatch.bench_adapter import make_pref_factory
from pref_dispatch.evaluate import DispatchController, rollout
from pref_dispatch.llm.basis import load_basis, load_frozen_combiner
from pref_dispatch.llm.reward_spec import describe_reward
from pref_dispatch.llm.sandbox import compile_reward
from pref_dispatch.nyc_env import TIME_OF_DAY
from pref_dispatch.preference import Preference
from ride_gym.rewards import DefaultRewardFunction

PREF_POINTS = (0.1, 0.5, 0.9)
# For an ignore_pref combiner the pref argument is dead, so any single point does;
# use neutral for the one capture rollout.
NEUTRAL = 0.5


def _pref(rev: float) -> Preference:
    return Preference(weights={"revenue": rev, "service": 1.0 - rev, "fairness": 0.0})


def _classify_state(driver_obs) -> str:
    """Bucket a driver into the interpretable states the combiner reasons about."""
    self_obs = driver_obs["self"]
    details = self_obs.get("assigned_order_details", []) or []
    if not details:
        return "idle/empty"
    etas = [d["eta"] for d in details if d.get("eta") is not None]
    # deadline-pressed = an onboard order with little slack; else loaded-with-slack.
    # (The combiner itself keys this off phi.mean_solo_time; here we only bucket for
    # display, so any short-eta onboard order counts as pressed.)
    if etas and min(etas) <= 6.0:
        return "deadline-pressed"
    return "loaded-with-slack"


def _representatives(samples) -> Dict[str, Tuple]:
    """One representative (driver_obs, phi) per state bucket from the capture."""
    reps: Dict[str, Tuple] = {}
    for driver_obs, phi in samples:
        st = _classify_state(driver_obs)
        reps.setdefault(st, (driver_obs, phi))
    return reps


def _reward_from_provenance(meta: Dict, env) -> Tuple[Optional[object], str]:
    """Reconstruct the TARGET reward this combiner was composed FOR.

    Returns ``(reward_function_or_None, description_text)``. For an authored reward
    the provenance carries the ``code`` (recompiled in the sandbox and adapted to
    the env's ``(driver_id, event)`` shape); for a given-instance / from-env reward
    we rebuild ``DefaultRewardFunction`` from the coefficient snapshot (or fall back
    to the env's own reward). ``None`` means "use the env's default reward as-is".
    """
    prov = meta.get("reward_provenance") or {}
    code = prov.get("code")
    if prov.get("authored") and code:
        reward_event_fn = compile_reward(code)

        def _env_reward(_driver_id, event, _fn=reward_event_fn):
            return float(_fn(event))

        text = (
            f"Authored reward: {prov.get('reward_name', '?')}\n"
            f"Objective: {prov.get('objective', '?')}\n\n"
            f"{prov.get('description', '')}\n\n"
            f"```python\n{code}\n```"
        )
        return _env_reward, text

    # Given-instance / from-env: rebuild DefaultRewardFunction from the snapshot so
    # the KPI row is graded by exactly the coefficients the combiner targeted.
    snap = meta.get("reward_snapshot") or prov.get("reward_snapshot")
    if snap:
        rf = DefaultRewardFunction(**{k: snap[k] for k in snap})
        return rf, describe_reward(rf)
    # No provenance: grade under the env's own reward (already installed).
    return None, describe_reward(env.reward_function)


def _report_single_objective(args, skills, skill_names, combiner, meta, base) -> None:
    """§Phase-2 single-reward report: one target reward -> one strategy -> one row."""
    # Reconstruct the target reward and inject it so income_mean IS its fleet mean.
    env_probe = make_benchmark_env(base)
    reward_fn, reward_text = _reward_from_provenance(meta, env_probe)
    env = make_benchmark_env(base, reward_function=reward_fn)
    env.reset(seed=args.seed)

    print("#" * 72)
    print("# FIXED REWARD -> ONE COMPOSED STRATEGY -> CONCRETE BEHAVIOUR "
          "(no preference dial)")
    print("#" * 72)
    print(f"\nanchor: {base.num_drivers} drivers / cap{base.driver_capacity} / "
          f"speed{base.speed_kmh:g} / {args.regime}-{args.split}"
          f"{'' if args.order_limit is None else f' / order_limit={args.order_limit}'}")
    print(f"combiner: {args.combiner}  (frozen skills: {list(skill_names)})")
    print("mode    : single-objective (ignore_pref) -- one reward, one strategy")

    # ---- 1. THE TARGET REWARD ------------------------------------------ #
    print("\n" + "=" * 72)
    print("1. THE TARGET REWARD (the fixed objective this strategy maximises)")
    print("=" * 72)
    print(reward_text)

    # ---- 2. THE AUTHOR'S / COMBINER'S UNDERSTANDING -------------------- #
    prov = meta.get("reward_provenance") or {}
    print("\n" + "=" * 72)
    print("2. REWARD UNDERSTANDING (LLM CoT gate)")
    print("=" * 72)
    if prov.get("reward_understanding"):
        print("author  :", prov["reward_understanding"])
    if meta.get("reward_understanding"):
        print("combiner:", meta["reward_understanding"])
    if not prov.get("reward_understanding") and not meta.get("reward_understanding"):
        print("(no recorded reward_understanding)")

    # ---- 3. THE COMPOSED STRATEGY -------------------------------------- #
    print("\n" + "=" * 72)
    print("3. THE COMPOSED STRATEGY")
    print("=" * 72)
    print("strategy   :", meta.get("strategy", "?"))
    print("description:", meta.get("description", "?"))

    # ---- 4a. FLEET SKILL MIX (single distribution -- pref is ignored) --- #
    combiner.enable_capture(1000)
    ctrl = DispatchController(combiner, skills=skills)
    metrics = rollout(env, ctrl, _pref(NEUTRAL), seed=args.seed)
    samples = list(combiner._obs_samples)

    print("\n" + "=" * 72)
    print(f"4a. FLEET SKILL MIX (single objective; captured fleet n={len(samples)})")
    print("=" * 72)
    frac = combiner.fleet_pick_fractions(_pref(NEUTRAL))
    for n in skill_names:
        print(f"  {n:>18} : {frac.get(n, 0.0):.3f}")
    print("\n(reading: the fraction of the fleet each frozen skill is assigned to "
          "\n maximise THIS reward. There is no slider -- one objective, one mix.)")

    # ---- 4b. PER-DRIVER PICKS (single column) -------------------------- #
    print("\n" + "=" * 72)
    print("4b. PER-DRIVER PICK for representative driver states")
    print("=" * 72)
    reps = _representatives(samples)
    if not reps:
        print("(no drivers captured -- try a larger --order-limit or full hour)")
    else:
        for st in ("idle/empty", "loaded-with-slack", "deadline-pressed"):
            if st not in reps:
                continue
            driver_obs, phi = reps[st]
            pick = combiner.argmax_pick(driver_obs, phi, _pref(NEUTRAL))
            print(f"  {st:<19} -> {pick}")
        # Prove pref is truly ignored: the pick is invariant to the pref argument.
        st0 = next(iter(reps))
        d0, p0 = reps[st0]
        same = (combiner.argmax_pick(d0, p0, _pref(0.1))
                == combiner.argmax_pick(d0, p0, _pref(0.9)))
        print(f"\n(ignore_pref check: pick(rev=0.1) == pick(rev=0.9) for {st0!r}: {same})")

    # ---- 4c. KPI ROW under the target reward --------------------------- #
    if not args.no_kpi:
        print("\n" + "=" * 72)
        print("4c. KPI under the TARGET reward (single objective; anchor-comparable)")
        print("=" * 72)
        print(f"  reward (income_mean) : {metrics['income_mean']:.4f}")
        print(f"  service_rate         : {metrics['service_rate']:.4f}")
        print(f"  completed            : {metrics['completed']:.0f}")
        print(f"  mean_service_time    : {metrics['mean_service_time']:.4f}")
        print(f"  detour_total         : {metrics['detour_total']:.4f}")
        print(f"  income_gini          : {metrics['income_gini']:.4f}")
        print("\n(reward is the fleet-mean cumulative value of the TARGET reward the "
              "\n strategy was composed to maximise -- the single number for this "
              "objective.)")


def _report_pref_slider(args, skills, skill_names, combiner, meta, base) -> None:
    """Legacy v2 report: behaviour across the runtime preference slider."""
    env = make_benchmark_env(base)
    env.reset(seed=args.seed)

    print("#" * 72)
    print("# GIVEN THE CURRENT REWARD FUNCTION -> COMPOSED STRATEGY -> BEHAVIOUR")
    print("#" * 72)
    print(f"\nanchor: {base.num_drivers} drivers / cap{base.driver_capacity} / "
          f"speed{base.speed_kmh:g} / {args.regime}-{args.split}"
          f"{'' if args.order_limit is None else f' / order_limit={args.order_limit}'}")
    print(f"combiner: {args.combiner}  (frozen skills: {list(skill_names)})")

    print("\n" + "=" * 72)
    print("1. THE REWARD FUNCTION (live coefficients off the env)")
    print("=" * 72)
    print(describe_reward(env.reward_function))
    snap = meta.get("reward_snapshot")
    if snap:
        print("\n[composed FOR reward coefficients snapshot]:", snap)

    print("\n" + "=" * 72)
    print("2. THE COMBINER'S UNDERSTANDING OF THE REWARD (LLM CoT gate)")
    print("=" * 72)
    print(meta.get("reward_understanding")
          or "(this combiner has no recorded reward_understanding -- was it evolved "
             "with --reward-conditioned?)")

    print("\n" + "=" * 72)
    print("3. THE COMPOSED STRATEGY")
    print("=" * 72)
    print("strategy   :", meta.get("strategy", "?"))
    print("description:", meta.get("description", "?"))

    combiner.enable_capture(1000)
    ctrl = DispatchController(combiner, skills=skills)
    _neutral_metrics = rollout(env, ctrl, _pref(0.5), seed=args.seed)
    samples = list(combiner._obs_samples)

    print("\n" + "=" * 72)
    print(f"4a. FLEET SKILL MIX across the preference slider "
          f"(captured fleet n={len(samples)})")
    print("=" * 72)
    header = "  rev  | " + "  ".join(f"{n:>16}" for n in skill_names)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for rev in PREF_POINTS:
        frac = combiner.fleet_pick_fractions(_pref(rev))
        row = "  ".join(f"{frac.get(n, 0.0):>16.3f}" for n in skill_names)
        print(f"  {rev:>4.1f} | {row}")
    print("\n(reading: how the fraction of the fleet choosing each frozen skill "
          "\n slides as the platform dials revenue up -- a smooth slide, not a snap, "
          "\n is the zero-retrain-adaptation headline.)")

    print("\n" + "=" * 72)
    print("4b. PER-DRIVER PICKS for representative driver states")
    print("=" * 72)
    reps = _representatives(samples)
    if not reps:
        print("(no drivers captured -- try a larger --order-limit or full hour)")
    else:
        print("  state               | " + "  ".join(f"rev={r:g}" for r in PREF_POINTS))
        print("  " + "-" * 60)
        for st in ("idle/empty", "loaded-with-slack", "deadline-pressed"):
            if st not in reps:
                continue
            driver_obs, phi = reps[st]
            picks = [combiner.argmax_pick(driver_obs, phi, _pref(r)) for r in PREF_POINTS]
            print(f"  {st:<19} | " + "  ".join(f"{p:<7}" for p in picks))
        print("\n(reading: deadline-pressed cars should hold a protect-onboard skill "
              "\n regardless of pref; idle cars should slide from service- to "
              "\n revenue-leaning skills as rev rises.)")

    if not args.no_kpi:
        print("\n" + "=" * 72)
        print("4c. KPI ROW under the actual reward (per preference; anchor-comparable)")
        print("=" * 72)
        methods = {
            f"{args.combiner}@rev{rev:g}":
                make_pref_factory(_pref(rev), combiner_name=args.combiner)
            for rev in PREF_POINTS
        }
        ts = TestSet(name=f"reward_strategy_{args.regime}_{args.split}",
                     seed=args.seed, split=None)
        evaluate(methods, [ts], base_cfg=base, out_dir=None, verbose=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--combiner", default="reward_maximizing_dispatcher",
                    help="frozen combiner name under evolved/combiners/.")
    ap.add_argument("--regime", default="peak", choices=sorted(TIME_OF_DAY))
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--order-limit", type=int, default=None,
                    help="cap orders (None = full hour; e.g. 400 for a quick look).")
    ap.add_argument("--no-kpi", action="store_true",
                    help="skip the KPI rollout (behaviour probes only, faster).")
    args = ap.parse_args()

    skills, _cards = load_basis(include_evolved=True)
    skill_names = tuple(skills)
    combiner, meta = load_frozen_combiner(args.combiner, skill_names=skill_names)

    # Build the anchor cfg exactly as the anchor table does (byte-identical stream).
    base = anchor_cfg(args.regime, args.split, args.order_limit)

    # Dispatch on the frozen combiner's own record: the §Phase-2 single-reward arm
    # marks itself ignore_pref, and gets the single-objective report; everything
    # else keeps the legacy preference-slider report.
    if meta.get("ignore_pref"):
        _report_single_objective(args, skills, skill_names, combiner, meta, base)
    else:
        _report_pref_slider(args, skills, skill_names, combiner, meta, base)


if __name__ == "__main__":
    main()
