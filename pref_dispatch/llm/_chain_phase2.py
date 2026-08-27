"""Wait for a running Phase 1 to finish, then auto-launch Phase 2.

Phase 2 trains the objective-reading combiner over the FROZEN Phase-1 skill
repository (``load_basis`` reads ``evolved/skills/``), so it may only start after
Phase 1 actually froze its skills. This watcher polls the Phase-1 pid, and once it
exits decides whether to chain:

  * Phase 1 must have reached its summary banner AND left frozen skills on disk --
    a crashed or gateway-starved run must NOT silently train a combiner over a
    half-built (or empty) repository;
  * the gateway must be healthy (the same real-training-size probe as
    ``_proxy_probe``, since the outages traced to egress IP, not to code). An
    unhealthy gateway is retried for a bounded window rather than failing at once;
  * no Phase 2 may already be running, so a second watcher cannot double-launch.

Every decision is appended to ``cache/logs/chain_phase2.log``.

Usage (detached, from the launcher below or by hand):
    python -m pref_dispatch.llm._chain_phase2 --phase1-pid 23036 \
        --phase1-log cache/logs/phase1_v4div_auto_20260808_175318.log

Exit codes: 0 = Phase 2 launched, 1 = not chained (reason in the log), 2 = gave up
waiting for a healthy gateway.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time
from typing import Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHAIN_LOG = os.path.join(ROOT, "cache", "logs", "chain_phase2.log")
SKILL_DIR = os.path.join(ROOT, "pref_dispatch", "evolved", "skills")
PHASE1_BANNER = "=== PHASE 1 SKILL REPOSITORY ==="


def _log(line: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(CHAIN_LOG), exist_ok=True)
    with open(CHAIN_LOG, "a", encoding="utf-8") as f:
        f.write(f"{stamp} {line}\n")
    print(f"{stamp} {line}", flush=True)


def _pid_alive(pid: int) -> bool:
    """True while ``pid`` is a live python process. Fails SAFE (alive on error) so a
    transient query failure never chains Phase 2 on top of a running Phase 1."""
    ps = ("powershell", "-NoProfile", "-Command",
          f"if (Get-Process -Id {pid} -ErrorAction SilentlyContinue) "
          "{ 'ALIVE' } else { 'GONE' }")
    try:
        out = subprocess.run(ps, capture_output=True, text=True, timeout=30)
        return not out.stdout.strip().endswith("GONE")
    except Exception:  # noqa: BLE001 -- unknown => treat as still running
        return True


def _module_running(module: str) -> bool:
    """True if any python process runs ``module``. Fails safe (unknown => running)."""
    # Exclude THIS process: the watcher's own command line mentions the module it
    # is about to launch, which would otherwise read as "already running".
    me = os.getpid()
    ps = ("powershell", "-NoProfile", "-Command",
          "$c = (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\") | "
          f"Where-Object {{ $_.CommandLine -like '*{module}*' -and "
          f"$_.ProcessId -ne {me} }}; "
          "if ($c) { 'RUNNING' } else { 'IDLE' }")
    try:
        out = subprocess.run(ps, capture_output=True, text=True, timeout=30)
        return out.stdout.strip().endswith("RUNNING")
    except Exception:  # noqa: BLE001
        return True


def _n_frozen_skills() -> int:
    """Frozen skills visible to ``load_basis`` (FLAT glob; subdirs are discarded)."""
    return len(glob.glob(os.path.join(SKILL_DIR, "*.meta.json")))


def _phase1_succeeded(log_path: Optional[str]) -> Tuple[bool, str]:
    n = _n_frozen_skills()
    if n == 0:
        return False, "no frozen skills in evolved/skills (repository empty)"
    if not log_path:
        return True, f"{n} frozen skills (no log given, banner unchecked)"
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            body = f.read()
    except OSError as e:
        return False, f"cannot read phase1 log: {e}"
    if PHASE1_BANNER not in body:
        return False, (f"phase1 log has no summary banner ({n} skills frozen) -- "
                       "run died or was stopped mid-way")
    return True, f"{n} frozen skills + summary banner present"


def _gateway_healthy() -> Tuple[bool, str]:
    """Real-training-size probe (same path as ``_proxy_probe``): a toy prompt would
    pass during the degraded windows that actually break training."""
    from pref_dispatch.llm._proxy_probe import _probe
    try:
        return _probe()
    except Exception as e:  # noqa: BLE001 -- watcher must never die on a probe error
        return False, f"probe raised {type(e).__name__}: {str(e)[:120]}"


def _launch_phase2(objectives: int, generations: int) -> Tuple[int, str]:
    log_path = os.path.join(ROOT, "cache", "logs",
                            "phase2_v3_auto_" + time.strftime("%Y%m%d_%H%M%S") + ".log")
    log_f = open(log_path, "ab", buffering=0)
    cmd = [sys.executable, "-u", "-m", "pref_dispatch.llm.run_phase2_full",
           "--objectives", str(objectives), "--generations", str(generations)]
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) \
        | getattr(subprocess, "DETACHED_PROCESS", 0)
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=log_f, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, creationflags=flags)
    return p.pid, log_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase1-pid", type=int, required=True)
    ap.add_argument("--phase1-log", default=None,
                    help="phase1 log; checked for the summary banner.")
    ap.add_argument("--objectives", type=int, default=8)
    ap.add_argument("--generations", type=int, default=5)
    ap.add_argument("--poll", type=float, default=120.0,
                    help="seconds between liveness polls.")
    ap.add_argument("--health-retries", type=int, default=12,
                    help="gateway probes before giving up (600s apart).")
    args = ap.parse_args()

    _log(f"watching phase1 pid={args.phase1_pid} (poll {args.poll:.0f}s); "
         f"will chain phase2 --objectives {args.objectives} "
         f"--generations {args.generations}")

    while _pid_alive(args.phase1_pid):
        time.sleep(args.poll)
    _log(f"phase1 pid={args.phase1_pid} exited")

    ok, why = _phase1_succeeded(args.phase1_log)
    if not ok:
        _log(f"NOT CHAINING: {why}")
        return 1
    _log(f"phase1 looks complete: {why}")

    if _module_running("run_phase2_full"):
        _log("NOT CHAINING: a phase2 run is already active")
        return 1

    # The gateway outages were IP-bound and self-healing; wait rather than abort.
    for attempt in range(1, args.health_retries + 1):
        healthy, detail = _gateway_healthy()
        if healthy:
            _log(f"gateway HEALTHY ({detail})")
            break
        _log(f"gateway UNHEALTHY ({detail}); attempt {attempt}/{args.health_retries}")
        if attempt == args.health_retries:
            _log("GAVE UP: gateway never recovered; phase2 not launched")
            return 2
        time.sleep(600.0)

    pid, log_path = _launch_phase2(args.objectives, args.generations)
    _log(f"LAUNCHED phase2 pid={pid} -> {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
