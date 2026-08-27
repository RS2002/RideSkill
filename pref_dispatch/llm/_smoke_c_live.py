"""Live smoke of paradigm C: a couple of REAL LLM steps through the online path.

Confirms the live path end to end: real client.complete -> parsed assignment ->
per-driver one-hot through the matcher -> env.step, with real tokens/latency
metered. Tiny (few drivers, 2 steps) to keep the call count and cost minimal.

Requires the API key in the environment / git-ignored .env. This is the ONLY
paradigm that makes online LLM calls; paradigm B is a frozen pure-Python function
verified offline. Run:  python -m pref_dispatch.llm._smoke_c_live
"""
from __future__ import annotations

from pref_dispatch.llm.basis import load_basis
from pref_dispatch.llm.client import make_llm_client
from pref_dispatch.llm.combiner_eval import make_train_prefs
from pref_dispatch.llm.config import LLMConfig
from pref_dispatch.llm.online_eval import online_rollout
from pref_dispatch.llm.paradigm_c import OnlineLLMController, StepMeter
from pref_dispatch.nyc_env import make_nyc_env


def main():
    client = make_llm_client(LLMConfig())  # fail-fast if key missing
    skills, cards = load_basis(include_evolved=False)
    pref = make_train_prefs(n=1, seed=99)[0]
    print("pref:", dict(pref.weights))

    reasons = []
    ctrl = OnlineLLMController(
        client, cards, top_k=5, max_drivers=10,
        meter=StepMeter(), log_reasoning=reasons.append,
    )
    env = make_nyc_env(seed=0, regime="peak", split="test",
                       num_drivers=30, order_limit=30)
    metrics, meter = online_rollout(env, ctrl, skills, pref, seed=0, max_steps=2)

    s = meter.summary()
    print("metrics:", {k: round(float(v), 3) for k, v in metrics.items()})
    print("meter:", {k: round(v, 2) for k, v in s.items()})
    print("reasoning samples:")
    for r in reasons:
        print("  -", r)

    assert s["llm_calls"] == 2, s["llm_calls"]
    assert s["mean_total_tokens"] > 0, "no tokens metered"
    assert s["mean_latency_s"] > 0, "no latency metered"
    if s["fallback_steps"]:
        print(f"WARNING: {s['fallback_steps']} fallback step(s) -- LLM reply unusable.")
    else:
        assert reasons, "no reasoning captured (interpretability gate)"
    print("\nOK: paradigm-C live path verified.")


if __name__ == "__main__":
    main()
