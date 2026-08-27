"""One-command Phase-3 quick fine-tune (§Feature 1): specialise the generalist
policy to a (possibly PARTIAL) operating point and report the before/after gain.

The operating point may be FULLY concrete OR partially specified: pass only the
axes you care about (e.g. just ``--num-drivers`` + a reward), and every axis you
LEAVE OUT (capacity / regime / speed / preference) is DOMAIN-RANDOMIZED during the
fine-tune -- exactly like the Phase-2 generalist evolution, only with your given
axes pinned. Use ``--n-scenarios N`` to control how many randomized draws the
fine-tune evolves over (default 1; use more when several axes are random).

Given the spec, this:

1. warm-start fine-tunes the frozen combiner + only the skills it actually uses on
   that batch (see :func:`pref_dispatch.llm.finetune.finetune_to_spec`), freezing
   the specialised artifacts to ``pref_dispatch/evolved/finetuned/<tag>/`` -- the
   generalist basis on disk is NEVER touched;
2. grades, on the SAME batch, the generalist ``ours(base)`` vs the fine-tuned
   ``ours(finetuned)`` vs the four point-free heuristics, under the same reward the
   combiner was composed for; and
3. writes ``cache/finetune/<tag>/results.{json,csv,md}`` with a KPI table + the
   fine-tune net-gain row.

Requires the API key in ``YIBU_API_KEY`` / a git-ignored ``.env`` (never in repo):

    # Fully concrete point (all axes pinned):
    python -m pref_dispatch.llm.run_finetune \\
        --num-drivers 1500 --driver-capacity 6 --speed-kmh 40 --regime peak \\
        --gens-combiner 2 --gens-skill 1 --lam 1

    # PARTIAL: pin only the fleet, randomize capacity/speed/regime/preference over
    # 8 sampled scenarios, under the MARL default-reward combiner:
    python -m pref_dispatch.llm.run_finetune \\
        --num-drivers 1000 --base-combiner default_reward_maximizer \\
        --n-scenarios 8 --gens-combiner 2 --gens-skill 1

Use ``--gens-combiner 0 --gens-skill 0`` for a fast wiring run (warm-start seed only,
no improve iterations -- fine-tuned == generalist, so the report should show ~0 gain).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Optional

from benchmark.evaluate import BASELINE_FACTORIES, MethodFactory, TestSet, _KPI_COLS, evaluate
from pref_dispatch.bench_adapter import make_pref_factory
from pref_dispatch.llm.finetune import ScenarioSpec, finetune_to_spec
from pref_dispatch.llm.sandbox import compile_reward
from pref_dispatch.scenario import Scenario


def _reward_from_provenance(reward_provenance: Optional[Dict]):
    """Reconstruct the authored ``(driver_id, event) -> float`` reward for grading.

    Mirrors :func:`pref_dispatch.llm.finetune._reward_from_combiner_meta` / the
    objective sweep: recompile the sandboxed body and adapt to the env call shape.
    Returns ``(reward_fn, reward_name)`` or ``(None, None)`` for a scalarize combiner.
    """
    prov = reward_provenance or {}
    code = prov.get("code")
    if not (prov.get("authored") and code):
        return None, None
    reward_event_fn = compile_reward(code)

    def _env_reward(_driver_id, event, _fn=reward_event_fn):
        return float(_fn(event))

    return _env_reward, prov.get("reward_name", "?")


def _persist(out_root: str, ts_name: str, summaries: Dict[str, Dict],
             meta: Dict) -> None:
    os.makedirs(out_root, exist_ok=True)
    cols = [k for k, _lbl, _w in _KPI_COLS]

    with open(os.path.join(out_root, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summaries": summaries}, f, indent=2)

    with open(os.path.join(out_root, "results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method"] + cols)
        for name, s in summaries.items():
            w.writerow([name] + [s.get(k, "") for k in cols])

    # Markdown: KPI table + fine-tune net-gain row.
    lines: List[str] = []
    lines.append(f"# Phase-3 fine-tune report: `{ts_name}`\n")
    lines.append(f"**Base combiner:** `{meta['base_combiner_name']}` -> "
                 f"**fine-tuned:** `{meta['finetuned_combiner_name']}`  ")
    lines.append(f"**Objective:** {meta['objective']}"
                 + (f" (reward `{meta['reward_name']}`)" if meta.get("reward_name") else "")
                 + "  ")
    lines.append(f"**Fine-tuned skills:** {meta['finetuned_skills'] or '(none)'} "
                 f"(selected: {meta['selected_skills']})  ")
    lines.append(f"**Combiner fitness on scenario:** base {meta['base_fitness']:.4g} -> "
                 f"fine-tuned {meta['finetuned_fitness']:.4g}\n")

    headers = ["method"] + [lbl for _k, lbl, _w in _KPI_COLS]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    # Sort by reward (the graded objective) descending.
    ordered = sorted(summaries.items(),
                     key=lambda kv: kv[1].get("total_reward", float("-inf")),
                     reverse=True)
    for name, s in ordered:
        row = [name]
        for k, _lbl, _w in _KPI_COLS:
            v = s.get(k)
            row.append(f"{v:.4g}" if isinstance(v, (int, float)) else str(v or ""))
        lines.append("| " + " | ".join(row) + " |")

    # Net gain: fine-tuned vs generalist on the graded reward.
    base_r = summaries.get(meta["base_method"], {}).get("total_reward")
    ft_r = summaries.get(meta["finetuned_method"], {}).get("total_reward")
    if base_r not in (None, 0) and ft_r is not None:
        margin = (ft_r - base_r) / abs(base_r) * 100.0
        lines.append(f"\n**Fine-tune net gain (env reward):** {base_r:.4g} -> {ft_r:.4g} "
                     f"= **{margin:+.1f}%** on this scenario.")
        if meta["objective"] == "scalarize":
            fit_margin = ((meta["finetuned_fitness"] - meta["base_fitness"])
                          / abs(meta["base_fitness"]) * 100.0
                          if meta["base_fitness"] else float("nan"))
            lines.append(
                "\n> Note: the `scalarize` objective optimises the preference-weighted "
                "multi-objective (revenue/service/gini), NOT raw env reward. The "
                f"optimised objective moved {meta['base_fitness']:.4g} -> "
                f"{meta['finetuned_fitness']:.4g} (**{fit_margin:+.1f}%**); the env-reward "
                "column above is a secondary view and need not move with it."
            )
    lines.append("")
    with open(os.path.join(out_root, "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-3 quick fine-tune to an operating point.")
    # Operating point. Any axis left UNSET (None) is domain-randomized during the
    # fine-tune; any axis given is pinned across the whole sampled batch.
    ap.add_argument("--num-drivers", type=int, default=None,
                    help="pin fleet size; omit to randomize it.")
    ap.add_argument("--driver-capacity", type=int, default=None,
                    help="pin vehicle capacity; omit to randomize it.")
    ap.add_argument("--speed-kmh", type=float, default=None,
                    help="pin cruise speed; omit to randomize it.")
    ap.add_argument("--regime", default=None, choices=["offpeak", "shoulder", "peak"],
                    help="pin the demand regime; omit to randomize it.")
    ap.add_argument("--pref-revenue", type=float, default=None,
                    help="pin the revenue preference weight; omit to randomize it "
                         "(ignored anyway when the base combiner is ignore_pref).")
    ap.add_argument("--split", default="test")
    ap.add_argument("--order-limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-scenarios", type=int, default=1,
                    help="how many randomized scenarios to fine-tune over when some "
                         "axes are unspecified (default 1).")
    # Fine-tune knobs.
    ap.add_argument("--base-combiner", default="state_aware_pref_slider")
    ap.add_argument("--gens-combiner", type=int, default=2)
    ap.add_argument("--gens-skill", type=int, default=1)
    ap.add_argument("--lam", type=int, default=1)
    ap.add_argument("--skill-pick-threshold", type=float, default=0.01)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--model", default=None, help="override LLMConfig.model")
    ap.add_argument("--out-root", default=os.path.join("cache", "finetune"))
    ap.add_argument("--no-compare", action="store_true",
                    help="skip the before/after benchmark table (fine-tune only).")
    args = ap.parse_args()

    spec = ScenarioSpec(
        num_drivers=args.num_drivers,
        driver_capacity=args.driver_capacity,
        speed_kmh=args.speed_kmh,
        regime=args.regime,
        pref_revenue=args.pref_revenue,
        order_limit=args.order_limit,
        split=args.split,
        seed=args.seed,
        n_scenarios=args.n_scenarios,
    )
    if spec.randomized:
        print(f"=== Phase-3 fine-tune | pinned={spec.pinned or '(none)'} "
              f"| RANDOMIZED={spec.randomized} over {spec.n_scenarios} scenario(s) ===")
    else:
        print(f"=== Phase-3 fine-tune to concrete point {spec.pinned} ===")

    res = finetune_to_spec(
        spec,
        base_combiner_name=args.base_combiner,
        model=args.model,
        generations_combiner=args.gens_combiner,
        generations_skill=args.gens_skill,
        lam=args.lam,
        skill_pick_threshold=args.skill_pick_threshold,
        temperature=args.temperature,
    )

    print("\n=== FINE-TUNE SUMMARY ===")
    print("tag           :", res.tag)
    print("pinned axes   :", res.pinned or "(none)")
    print("randomized    :", res.randomized_axes or "(none)")
    print("scenarios     :", res.n_scenarios)
    print("selected skill:", res.selected_skills)
    print("finetuned skl :", res.finetuned_skills)
    print("combiner name :", res.combiner_name)
    print("skills dir    :", res.skills_dir)
    print("combiners dir :", res.combiners_dir)
    print("combiner path :", res.combiner_path)
    print("fitness       : base %.4g -> finetuned %.4g" % (
        res.base_fitness, res.finetuned_fitness))

    if args.no_compare:
        return

    # --- Before/after comparison on the SAME batch. ----------------------- #
    base_method = f"ours(base:{res.base_combiner_name})"
    ft_method = f"ours(finetuned:{res.combiner_name})"
    # The graded reward must match what the fine-tune composed for. Reconstruct the
    # authored reward if present; otherwise None = the env's own default reward
    # (matching an env-reward / ignore_pref base combiner such as
    # default_reward_maximizer), or a scalarize combiner graded on env reward as a
    # secondary view.
    reward_fn, reward_name = _reward_from_provenance(res_reward_provenance(res))

    methods: Dict[str, MethodFactory] = dict(BASELINE_FACTORIES)
    methods[base_method] = make_pref_factory(
        res.scenario.preference, combiner_name=res.base_combiner_name)
    methods[ft_method] = make_pref_factory(
        res.scenario.preference, combiner_name=res.combiner_name,
        skill_dir_override=[res.skills_dir] if res.finetuned_skills else None,
        combiner_dir_override=res.combiners_dir,
    )

    out_root = os.path.join(args.out_root, res.tag)
    # Grade on the SAME scenario batch the fine-tune evolved over. Each scenario
    # pins its own fleet/capacity/speed/window via to_config(), so we grade it as
    # its own single-scenario evaluate() call and aggregate across the batch.
    print(f"\n=== BEFORE/AFTER on {res.tag} over {len(res.scenarios)} scenario(s) "
          f"(reward={reward_name or 'default'}) ===")
    all_results: Dict[str, Dict[str, Dict]] = {}
    for i, sc in enumerate(res.scenarios):
        ts = TestSet(name=f"{res.tag}__s{i}", seed=sc.seed)
        r = evaluate(
            methods, [ts], base_cfg=sc.to_config(),
            out_dir=None, reward_function=reward_fn,
        )
        all_results[ts.name] = r[ts.name]
    # Aggregate KPI columns across the batch (mean) for the report.
    summaries = _aggregate_summaries(all_results)

    meta = {
        "tag": res.tag,
        "pinned": res.pinned,
        "randomized_axes": res.randomized_axes,
        "n_scenarios": res.n_scenarios,
        "scenarios": [sc.label() for sc in res.scenarios],
        "base_combiner_name": res.base_combiner_name,
        "finetuned_combiner_name": res.combiner_name,
        "base_method": base_method,
        "finetuned_method": ft_method,
        "objective": res.objective,
        "reward_name": reward_name,
        "selected_skills": res.selected_skills,
        "finetuned_skills": res.finetuned_skills,
        "base_fitness": res.base_fitness,
        "finetuned_fitness": res.finetuned_fitness,
    }
    _persist(out_root, res.tag, summaries, meta)
    print(f"\nwritten -> {os.path.join(out_root, 'results.md')}")


def _aggregate_summaries(all_results: Dict[str, Dict[str, Dict]]) -> Dict[str, Dict]:
    """Mean each KPI column across the batch's per-scenario summaries, per method.

    ``evaluate`` returns ``{test_set_name: {method: {kpi: value}}}``. For the report
    we average each numeric KPI over the scenarios so the table is one row per method
    (a single concrete scenario passes through unchanged).
    """
    cols = [k for k, _lbl, _w in _KPI_COLS]
    per_method: Dict[str, List[Dict]] = {}
    for _ts, methods in all_results.items():
        for name, s in methods.items():
            per_method.setdefault(name, []).append(s)
    out: Dict[str, Dict] = {}
    for name, rows in per_method.items():
        agg: Dict[str, object] = {}
        for k in cols:
            vals = [r.get(k) for r in rows if isinstance(r.get(k), (int, float))]
            if vals:
                agg[k] = sum(vals) / len(vals)
        out[name] = agg
    return out


def res_reward_provenance(res) -> Optional[Dict]:
    """Read the fine-tuned combiner's reward_provenance from its frozen meta.

    The grading reward must match what the fine-tune composed for, so we read it
    back from the just-frozen combiner meta rather than re-deriving it.
    """
    meta_path = os.path.join(res.combiners_dir, f"{res.combiner_name}.meta.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f).get("reward_provenance")


if __name__ == "__main__":
    main()
