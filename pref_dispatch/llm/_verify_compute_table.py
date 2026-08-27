"""Offline verification of the B/C compute table with NO real LLM.

Freezes a trivial valid combiner to a temp dir (paradigm B), builds a fake client
that returns a fixed JSON assignment (paradigm C), and drives the exact code path
of compute_table.main() end to end on a tiny env. Confirms: B full + B@k +
C(online) all produce finite scalar effects, and the StepMeter records tokens and
latency. No API key required.
"""
from __future__ import annotations

import json
import os
import tempfile

from pref_dispatch.evaluate import DispatchController, rollout
from pref_dispatch.llm.basis import load_basis, load_frozen_combiner
from pref_dispatch.llm.combiner_eval import build_norm_frame, make_train_prefs
from pref_dispatch.llm.compute_table import _b_rollout_truncated
from pref_dispatch.llm.online_eval import online_rollout
from pref_dispatch.llm.paradigm_c import OnlineLLMController, StepMeter
from pref_dispatch.generalize import scalarize
from pref_dispatch.nyc_env import make_nyc_env


class FakeClient:
    """Returns a valid assignment JSON; exposes last_usage like APIClient."""

    def __init__(self):
        self.last_usage = None

    def complete(self, system, user, *, temperature=None):
        # Assign every driver id mentioned in the prompt the 'revenue' skill.
        import re
        ids = re.findall(r"driver (\d+) \[", user)
        assigns = {i: "revenue" for i in ids}
        self.last_usage = {"prompt_tokens": 1900, "completion_tokens": 40}
        return json.dumps({
            "assignments": assigns,
            "reasoning": "revenue-heavy step: send idle cars to long fares.",
        })


def _freeze_trivial_combiner(out_dir, skill_names):
    """Write a minimal valid skill_scores module + meta to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    name = "trivial_test_combiner"
    code = (
        "def skill_scores(driver_obs, phi, pref):\n"
        "    # Revenue-leaning when pref favours revenue, else service.\n"
        "    r = float(pref['revenue'])\n"
        "    return {'revenue': r, 'service': 1.0 - r, 'enroute': 0.2}\n"
    )
    with open(os.path.join(out_dir, f"{name}.py"), "w", encoding="utf-8") as f:
        f.write('"""trivial test combiner."""\nimport math\nimport numpy as np\n\n\n' + code)
    with open(os.path.join(out_dir, f"{name}.meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "combiner_name": name, "strategy": "test", "description": "test",
            "code": code, "skill_names": list(skill_names),
        }, f)
    return name


def main():
    regime, split = "peak", "test"
    seed, c_steps, n_prefs = 0, 4, 2
    num_drivers, order_limit = 40, 40  # tiny for speed

    skills, cards = load_basis(include_evolved=False)

    with tempfile.TemporaryDirectory() as tmp:
        name = _freeze_trivial_combiner(tmp, tuple(skills))
        combiner_b, meta = load_frozen_combiner(
            name, skill_names=tuple(skills), combiners_dir=tmp
        )
        print("loaded frozen B combiner:", meta["combiner_name"], "basis:", list(skills))

        def env_factory():
            return make_nyc_env(
                seed=seed, regime=regime, split=split,
                num_drivers=num_drivers, order_limit=order_limit,
            )

        prefs = make_train_prefs(n=n_prefs, seed=seed + 99)
        ranges = build_norm_frame(
            skills, prefs, regimes=(regime,), split=split,
            num_drivers=num_drivers, order_limit=order_limit, seed=seed,
        )

        client = FakeClient()
        meter_all = StepMeter()
        b_full, b_at_k, c_eff = [], [], []
        for pref in prefs:
            m_b = rollout(env_factory(), DispatchController(combiner_b, skills=skills), pref, seed=seed)
            b_full.append(scalarize(m_b, pref, ranges))
            m_bk = _b_rollout_truncated(env_factory, combiner_b, skills, pref, seed, c_steps)
            b_at_k.append(scalarize(m_bk, pref, ranges))
            ctrl = OnlineLLMController(client, cards, top_k=5, max_drivers=20, meter=meter_all)
            m_c, _ = online_rollout(env_factory(), ctrl, skills, pref, seed=seed, max_steps=c_steps)
            c_eff.append(scalarize(m_c, pref, ranges))

        s = meter_all.summary()
        print("B(full) effects:", [round(x, 3) for x in b_full])
        print("B@k     effects:", [round(x, 3) for x in b_at_k])
        print("C       effects:", [round(x, 3) for x in c_eff])
        print("meter:", {k: round(v, 2) for k, v in s.items()})

        assert all(x == x for x in b_full + b_at_k + c_eff), "non-finite effect"
        assert s["llm_calls"] == n_prefs * c_steps, s["llm_calls"]
        assert s["fallback_steps"] == 0, s["fallback_steps"]
        assert s["mean_total_tokens"] > 0
        print("\nOK: B/C compute-table path verified offline.")


if __name__ == "__main__":
    main()
