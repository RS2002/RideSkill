"""Minimal LIVE smoke: ONE real LLM call end-to-end (needs YIBU_API_KEY via .env).

Confirms the real pipeline works before committing to a full run:
  client -> prompt -> real LLM -> extract JSON -> sandbox compile+validate ->
  rollout on a tiny scenario -> finite rescaled fitness + NL explanation.

Runs Phase-1 step 1a for ONE direction only (run_self_invention=False,
generations=0, freeze=False) on a tiny capped scenario, so it is ~1-2 API calls
and a couple of fast rollouts. Prints the evolved skill's objective + score.
"""

from __future__ import annotations

import random
import time

from pref_dispatch.llm.client import make_llm_client
from pref_dispatch.llm.config import LLMConfig
from pref_dispatch.llm.run_phase1 import run_phase1
from pref_dispatch.scenario import ScenarioRanges, ScenarioSampler


def main() -> None:
    cfg = LLMConfig(temperature=0.7, max_tokens=2048)
    client = make_llm_client(cfg)
    print(f"[smoke] client built: provider={cfg.provider} model={cfg.model} "
          f"base_url={cfg.base_url}")

    # Tiny scenario envelope so the two rollouts are fast.
    ranges = ScenarioRanges(fleet=(120, 120), capacity=(4, 4),
                            speed_kmh=(35.0, 35.0), order_limit=150)
    sampler = ScenarioSampler(ranges=ranges, rng=random.Random(0), split="train")

    t0 = time.time()
    res = run_phase1(
        client,
        env_profile="# LIVE SMOKE ENV\nManhattan ride-pooling, ~120 cars, capacity 4, "
        "offpeak, capped orders.\n",
        directions=(
            "Serve reachable riders with small detours: prefer short-pickup, "
            "short-ride orders and keep passengers moving.",
        ),
        sampler=sampler,
        scenarios_per_round=2,
        n_sig_scenarios=2,
        run_self_invention=False,
        generations=0,          # gen-0 only: propose + evaluate, no improve loop
        lam=1,
        freeze=False,           # do not write artifacts in a smoke run
        seed=0,
        temperature=0.7,
    )
    dt = time.time() - t0

    assert res.n_directed == 1, f"expected 1 directed skill, got {res.n_directed}"
    b = res.directed[0]
    cand = b.candidate
    ev = cand.evaluation
    score = getattr(ev, "score", None)
    print(f"[smoke] LIVE OK in {dt:.1f}s")
    print(f"        skill: {b.name!r}")
    print(f"        objective: {b.objective}")
    print(f"        fitness_rationale: {cand.meta['fitness_rationale']}")
    print(f"        rescaled score: {score}")
    print(f"        last_usage: {getattr(client, 'last_usage', 'n/a')}")


if __name__ == "__main__":
    main()
