"""Probe the LLM gateway with a REAL training-size prompt; auto-resume Phase 1.

The Phase 1-3 training loops depend on ~10-17k-char code-generation prompts
(``build_skill_improve_prompt`` / ``build_skill_prompt`` with the full env
profile). The yibuapi gateway was observed (2026-08-07) returning EMPTY
completions for user prompts above ~4-6k chars while small requests stayed
healthy, and recovering later the same day -- so a faithful health check must
send a real training prompt, not a toy one.

This script builds the exact improve prompt the training loop sends (same
``make_nyc_env`` + ``encode_env_profile`` + ``build_skill_improve_prompt``
path as ``run_phase1_full``) and reports HEALTHY only when non-empty content
comes back. When healthy it optionally restarts the frozen Phase 1 training,
guarding against double-runs via the process table and against relaunching a
completed run via a small state file.

Usage:
    python -m pref_dispatch.llm._proxy_probe                # probe + auto-resume
    python -m pref_dispatch.llm._proxy_probe --probe-only   # probe only

Exit codes: 0 = healthy (no action needed / held), 1 = unhealthy,
            2 = Phase 1 launched. One line appended to cache/logs/proxy_probe.log.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time
from typing import Dict, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROBE_LOG = os.path.join(ROOT, "cache", "logs", "proxy_probe.log")
STATE_FILE = os.path.join(ROOT, "cache", "logs", "auto_resume_state.json")

# The incumbent / fitness / objective the probe evolves are representative of
# the real training calls (direction 2 of run_phase1_full). They only need to
# reproduce the prompt SIZE and CODE-GEN shape the gateway must serve.
_INCUMBENT = """def score(driver_obs, order, phi_ep, phi_step):
    if not _feasible(driver_obs, order):
        return -1e9
    scale = float(phi_step.mean_solo_time) or float(phi_ep.scale) or 1.0
    pickup = _pickup_time(driver_obs, order, phi_ep.dist) / scale
    ride = _solo_time(order, phi_ep.dist) / scale
    detour = max(0.0, float(order.get("extra_detour_time", 0.0))) / scale
    return -(pickup + ride + 2.0 * detour)""".strip()

_FITNESS = """def fitness(metrics):
    return 100.0 * metrics['service_rate'] - metrics['mean_service_time']"""

_OBJECTIVE = ("Minimise passenger waiting and in-car time: serve the riders who "
              "can be picked up and dropped off soonest with the smallest added detour.")


def _log(line: str) -> None:
    os.makedirs(os.path.dirname(PROBE_LOG), exist_ok=True)
    with open(PROBE_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _load_state() -> Dict:
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"last_probe_ok": False, "phase": "idle",
                "unhealthy_after_launch": False}


def _save_state(state: Dict) -> None:
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _probe() -> Tuple[bool, Optional[str]]:
    """Send a REAL training-size improve prompt; also check the bigger fresh
    prompt when the improve one already works (partial recovery would leave
    gen-0 calls failing even if improvements pass)."""
    from pref_dispatch.llm.client import make_llm_client
    from pref_dispatch.llm.config import LLMConfig
    from pref_dispatch.llm.encode import encode_env_profile
    from pref_dispatch.llm.prompts.skill_evolve import (
        build_skill_improve_prompt,
        build_skill_prompt,
    )
    from pref_dispatch.llm.run_phase1_full import make_nyc_env
    from pref_dispatch.global_stats import GlobalStats
    from pref_dispatch.scenario import ScenarioRanges

    env = make_nyc_env(seed=0, regime="peak", split="train",
                       num_drivers=800, order_limit=None)
    env.reset(seed=0)
    dist = lambda a, b: env.network.shortest_path(a, b).travel_time
    phi = GlobalStats.from_observations(env._build_observations(), dist=dist)
    ranges = ScenarioRanges(order_limit=None,
                            order_limits=(None, 2500, 4000, 6000, 8000),
                            fleet_dist="loguniform")
    profile = encode_env_profile(env, phi, "peak", "train",
                                 random.Random(0), dist=dist,
                                 ranges=ranges, prev_windows=(1, 2))

    improve = build_skill_improve_prompt(
        profile, objective=_OBJECTIVE, fitness_code=_FITNESS,
        current_code=_INCUMBENT, current_fitness=0.07906)
    fresh = build_skill_prompt(profile, objective_hint=_OBJECTIVE)

    # Bounded single-attempt call: healthy calls take 20-130s, the broken
    # gateway returns empty or hits the wall-clock -- either way the probe
    # returns in <= timeout+5s.
    cfg = LLMConfig()
    cfg.timeout = 150.0
    cfg.n_retry = 1
    client = make_llm_client(cfg)

    def call(prompt: Dict[str, str], tag: str) -> Tuple[bool, str]:
        t0 = time.time()
        try:
            r = client.complete(prompt["system"], prompt["user"], temperature=0.9)
            ok = len(r.strip()) > 0
            detail = f"{tag} len={len(r)}" if ok else f"{tag} empty completion"
            return ok, f"dt={time.time() - t0:.0f}s {detail}"
        except Exception as e:  # noqa: BLE001 -- probe reports any failure
            return False, f"{tag} {type(e).__name__}: {str(e)[:100]}"

    ok1, d1 = call(improve, "improve")
    if not ok1:
        # Diagnostic: during degraded windows, does splitting the user prompt
        # into small messages pass when the single large message fails? Pure
        # observation -- the launch decision below is unchanged. Distinguishes
        # a per-message size threshold (split helps) from a per-request total
        # threshold (split useless).
        diag = _split_diag(cfg, improve["system"], improve["user"])
        return False, d1 + " | " + diag
    ok2, d2 = call(fresh, "fresh")
    if not ok2:
        diag = _split_diag(cfg, fresh["system"], fresh["user"])
        return False, d2 + " | " + diag
    return True, d1 + " | " + d2


def _split_diag(cfg: "LLMConfig", system: str, user: str, n: int = 4) -> str:
    """Split ``user`` into ``n`` messages under the observed ~4-6k-char failure
    threshold and call the gateway once with the same OpenAI client shape.
    Returns a one-line result for the probe log; never raises."""
    from openai import OpenAI

    key = _read_key_safely(cfg)
    if key is None:
        return "split-4: no key"
    size = len(user) // n
    chunks = [user[i * size:(i + 1) * size] if i < n - 1
              else user[(n - 1) * size:] for i in range(n)]
    msgs = [{"role": "system", "content": system}] + \
           [{"role": "user", "content": ch} for ch in chunks]
    t0 = time.time()
    try:
        client = OpenAI(api_key=key, base_url=cfg.base_url, timeout=60.0)
        result: dict = {}

        def _attempt():
            try:
                resp = client.chat.completions.create(
                    model=cfg.model, messages=msgs, temperature=0.9,
                    max_tokens=cfg.max_tokens)
                result["content"] = resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001 -- diagnostic
                result["err"] = e

        # Same wall-clock bound as client.py: the SDK timeout does not fire
        # against this proxy, so cap the attempt with a daemon thread.
        t = threading.Thread(target=_attempt, name="split-diag", daemon=True)
        t.start()
        t.join(65.0)
        if t.is_alive():
            return "split-4: FAIL wall-clock 65s"
        if "content" in result and result["content"].strip():
            return f"split-4: len={len(result['content'])} " \
                   f"dt={time.time() - t0:.0f}s"
        err = result.get("err")
        return f"split-4: FAIL {type(err).__name__} {str(err)[:80]} " \
               f"dt={time.time() - t0:.0f}s"
    except Exception as e:  # noqa: BLE001 -- diagnostic reports any failure
        return f"split-4: FAIL {type(e).__name__} {str(e)[:80]} " \
               f"dt={time.time() - t0:.0f}s"


def _read_key_safely(cfg: "LLMConfig") -> "Optional[str]":
    """Env key without raising: the probe must never crash on a missing key."""
    import os
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except ImportError:
        pass
    return os.environ.get(cfg.api_key_env)


def _phase1_running() -> bool:
    """True if any python process is running ``run_phase1_full``. Fails safe
    (treats unknown as running) so we never double-launch training."""
    ps = ("powershell", "-NoProfile", "-Command",
          "$c = (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\") | "
          "Where-Object { $_.CommandLine -like '*run_phase1_full*' }; "
          "if ($c) { 'RUNNING' } else { 'IDLE' }")
    try:
        out = subprocess.run(ps, capture_output=True, text=True, timeout=30)
        return out.stdout.strip().endswith("RUNNING")
    except Exception:  # noqa: BLE001
        return True


def _launch_phase1() -> Tuple[int, str]:
    log_path = os.path.join(ROOT, "cache", "logs",
                            "phase1_v4div_auto_"
                            + time.strftime("%Y%m%d_%H%M%S") + ".log")
    log_f = open(log_path, "ab", buffering=0)
    cmd = [sys.executable, "-u", "-m", "pref_dispatch.llm.run_phase1_full",
           "--scenarios", "6", "--sig-scenarios", "4",
           "--generations", "5", "--max-skills", "10"]
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) \
        | getattr(subprocess, "DETACHED_PROCESS", 0)
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=log_f, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, creationflags=flags)
    return p.pid, log_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe-only", action="store_true",
                    help="probe only; never launch training.")
    args = ap.parse_args()

    ok, detail = _probe()
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    state = _load_state()

    if not ok:
        # Gateway down. Mark the state so a later recovery restarts a run that
        # died mid-way (unhealthy_after_launch) but never relaunches one that
        # completed while the gateway stayed healthy.
        state["last_probe_ok"] = False
        if state.get("phase") == "launched":
            state["unhealthy_after_launch"] = True
        _save_state(state)
        line = f"{stamp} UNHEALTHY ({detail})"
        _log(line)
        print(line, flush=True)
        return 1

    if args.probe_only:
        line = f"{stamp} HEALTHY ({detail}) [probe-only]"
        _log(line)
        print(line, flush=True)
        return 0

    if _phase1_running():
        line = f"{stamp} HEALTHY ({detail}) [phase1 already running; skip]"
        _save_state({**state, "last_probe_ok": True})
        _log(line)
        print(line, flush=True)
        return 0

    # Launch only on a recovery signal: previous probe unhealthy, or a
    # previously-launched run died while the gateway was down. A run that
    # completed while probes stayed healthy (last_probe_ok True) is never
    # relaunched.
    if state.get("last_probe_ok") and not state.get("unhealthy_after_launch"):
        line = f"{stamp} HEALTHY ({detail}) [stable healthy; hold]"
        _log(line)
        print(line, flush=True)
        return 0

    pid, log_path = _launch_phase1()
    state = {"last_probe_ok": True, "phase": "launched",
             "unhealthy_after_launch": False,
             "launched_at": stamp, "pid": pid}
    _save_state(state)
    line = f"{stamp} HEALTHY ({detail}) [LAUNCHED phase1 pid={pid} -> {log_path}]"
    _log(line)
    print(line, flush=True)
    return 2


if __name__ == "__main__":
    sys.exit(main())
