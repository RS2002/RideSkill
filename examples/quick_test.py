"""Quick test: load the frozen RideSkill stack and roll one held-out NYC hour.

No LLM access is needed -- the skills, combiner and repositioner are the frozen
artifacts under pref_dispatch/evolved/. The episode is graded by the benchmark's
default reward; pass your own `w` (see below) to test objective conditioning.

Run from the repository root:

    python examples/quick_test.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.config import BenchmarkConfig
from benchmark.runner import run_episode
from pref_dispatch.bench_adapter import make_pref_factory
from pref_dispatch.llm.basis import load_repositioner
from pref_dispatch.nyc_env import _window_for_hour
from pref_dispatch.preference import Preference
from pref_dispatch.reposition import Repositioner

COMBINER = "objective_shape_dispatcher_r4"
REPOSITIONER = "dualmech_od_reach_hybrid_scorer"

# A neutral platform preference (fairness off).
pref = Preference({"revenue": 0.5, "service": 0.5, "fairness": 0.0})

# The frozen Phase-3 repositioner, wrapped in the runtime handle.
scores_fn, _meta = load_repositioner(REPOSITIONER)
rep = Repositioner(strength=1.0, scores_fn=scores_fn)

# One held-out test hour: 18:00, fleet 1,000, capacity 4, 35 km/h.
cfg = BenchmarkConfig(
    network_kind="nyc", num_drivers=1000, driver_capacity=4, speed_kmh=35.0,
    nyc_splits_dir=None, nyc_order_path=_window_for_hour("18", "test"),
    nyc_order_limit=None, nyc_split="test", seed=0,
)

# Optional: hand the stack a custom objective `w`. The combiner probes it on
# synthetic events at runtime and re-routes the fleet -- no retraining. Any
# additive per-event price list works; None runs objective-blind.
w = None

factory = make_pref_factory(pref, combiner_name=COMBINER, repositioner=rep,
                            reward_fn=w, objective_label="quick_test")
agent = factory(cfg)

print("Rolling one full hour (first run also precomputes the road-network "
      "cache, which takes a few minutes) ...")
summary, _recorder = run_episode(agent, "rideskill", cfg=cfg, verbose=False)

print("\n=== RideSkill frozen stack -- one held-out NYC hour ===")
for key, label in [
    ("total_reward", "Reward"),
    ("service_rate", "Service rate"),
    ("complete_rate", "Complete rate"),
    ("avg_wait_time", "Wait (min)"),
    ("avg_ride_time", "Ride (min)"),
    ("avg_detour_time", "Detour (min)"),
    ("avg_driver_utilisation", "Utilisation"),
]:
    print(f"  {label:<16} {summary[key]:.4f}")
