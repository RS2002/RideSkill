"""Offline verification for the final-version two-layer stats (phi_ep / phi_step).

No LLM key needed. Rolls a small hand-written skill / combiner / repositioner
through the real ``rollout`` on one :class:`Scenario` and asserts the two-layer
contract the redesign introduced:

 (a) phi_ep is EPISODE-STATIC: constructed once at rollout start, it never changes
     across steps, and it carries the leak-free static scale + dist + region layout
     (no future-order info).
 (b) phi_step is LIVE: recomputed every step; at least one of its live aggregates
     (num_pending / demand_pressure / mean_solo_time / kappa) actually moves across
     the episode -- proving it is not accidentally frozen like phi_ep.
 (c) kappa on phi_step: per-region demand / supply arrays are present and
     region-length, seeded from the live observation each step.
 (d) w is carried on phi_ep and reaches the combiner + repositioner (but NOT the
     skills): a probe combiner/repositioner that records whether it saw ``w`` fires
     true; a probe skill that would record ``w`` never receives it (skills take only
     ``(driver_obs, order, phi_ep, phi_step)``).
 (e) end-to-end: the whole loop runs and returns finite metrics under the new
     signatures with an objective ``w`` supplied.

These checks pin the interface, not the policy quality (that is the sweeps' job).
"""
from __future__ import annotations

import numpy as np

from pref_dispatch.combiner import Combiner
from pref_dispatch.evaluate import DispatchController, rollout
from pref_dispatch.global_stats import EpisodeStats, GlobalStats
from pref_dispatch.preference import Preference
from pref_dispatch.reposition import Repositioner
from pref_dispatch.scenario import Scenario, build_env
from pref_dispatch.skills import Skill


_PREF = Preference({"revenue": 0.5, "service": 0.5, "fairness": 0.0})

# A trivial objective: reward = served trip-minutes. Callable reward fn (w).
def _w(event) -> float:
    return float(sum((event.get("assigned_solo_times") or {}).values()))


class _ProbeSkill(Skill):
    """A minimal always-feasible scorer that ASSERTS its call surface.

    Records the phi_ep object identity it saw so the caller can prove phi_ep is the
    same instance every step (episode-static). Skills must NOT receive ``w`` -- the
    signature has no slot for it, which is the structural guarantee.
    """

    name = "revenue"

    def __init__(self):
        self.seen_phi_ep = set()
        self.seen_phi_step = set()

    def score(self, driver_obs, order, phi_ep, phi_step):
        self.seen_phi_ep.add(id(phi_ep))
        self.seen_phi_step.add(id(phi_step))
        # Cheap positive score for any feasible order so the loop actually bids.
        s = driver_obs["self"]
        if order["num_passengers"] > s["capacity"] - s["committed_passengers"]:
            return -1e9
        return 1.0

    def noop_score(self, driver_obs, phi_ep, phi_step):
        return 0.5


class _ProbeCombiner(Combiner):
    """Single-skill combiner that records whether it ever received ``w``."""

    def __init__(self, skill_name="revenue"):
        self.skill_name = skill_name
        self.saw_w = False
        self.phi_ep_ids = set()
        self.phi_step_ids = set()

    def weights_for(self, driver_obs, phi_ep, phi_step, w=None):
        if w is not None:
            self.saw_w = True
        self.phi_ep_ids.add(id(phi_ep))
        self.phi_step_ids.add(id(phi_step))
        return {self.skill_name: 1.0}

    def classify(self, driver_obs, phi_ep, phi_step):
        return self.skill_name


class _ProbeReposScorer:
    """A callable reposition scorer recording that it saw kappa + w + two-layer phi."""

    def __init__(self):
        self.saw_w = False
        self.saw_kappa = False
        self.n_calls = 0

    def __call__(self, driver_obs, phi_ep, phi_step, kappa, w):
        self.n_calls += 1
        if w is not None:
            self.saw_w = True
        if kappa is not None and getattr(kappa, "eff_demand", None) is not None:
            self.saw_kappa = True
        # Defer to the built-in demand-gravity kernel (return {} = no opinion).
        return {}


def _small_scenario():
    return Scenario(num_drivers=60, driver_capacity=4, speed_kmh=35.0,
                    regime="peak", split="train", order_limit=120,
                    pref_revenue=0.5, seed=0)


def test_phi_ep_static_and_phi_step_live() -> None:
    """Instrument the loop by hand (mirrors rollout) to watch phi_ep vs phi_step."""
    from pref_dispatch.evaluate import _make_dist

    sc = _small_scenario()
    env = build_env(sc)
    obs, _ = env.reset(seed=0)
    dist = _make_dist(env)
    speed = float(getattr(getattr(env, "config", None), "vehicle_speed_kmh", 0.0) or 0.0)
    phi_ep = EpisodeStats.from_observations(
        obs, dist=dist, speed_kmh=speed, reward_fn=_w, objective_label="served-minutes"
    )

    # phi_ep snapshot fields we assert stay constant.
    ep0 = (phi_ep.num_drivers, phi_ep.driver_capacity, phi_ep.speed_kmh,
           phi_ep.scale, phi_ep.region_centres, phi_ep.region_neighbours)

    live_seen = []  # (num_pending, demand_pressure, mean_solo_time)
    kappa_lens = []
    steps = 0
    done = False
    skill = _ProbeSkill()
    comb = _ProbeCombiner()
    ctrl = DispatchController(comb, skills={"revenue": skill}, top_k=0)
    income = {d: 0.0 for d in obs}
    while not done and steps < 40:
        # Recompute phi_step by hand to inspect it (controller.act does the same).
        phi_step = GlobalStats.from_observations(obs, dist=phi_ep.dist)
        live_seen.append(
            (phi_step.num_pending, phi_step.demand_pressure, phi_step.mean_solo_time)
        )
        kappa_lens.append((len(phi_step.region_demand), len(phi_step.region_supply)))
        actions = ctrl.act(obs, _PREF, income, phi_ep,
                           fairness_income={d: 0.0 for d in obs})
        obs, rewards, dones, info = env.step(actions)
        for d, r in rewards.items():
            income[d] = income.get(d, 0.0) + float(r)
        done = dones["__all__"]
        steps += 1

    # (a) phi_ep never mutated (frozen dataclass) and its snapshot is unchanged.
    ep1 = (phi_ep.num_drivers, phi_ep.driver_capacity, phi_ep.speed_kmh,
           phi_ep.scale, phi_ep.region_centres, phi_ep.region_neighbours)
    assert ep0 == ep1, (ep0, ep1)
    # The skill saw exactly one phi_ep identity across the whole episode.
    assert len(skill.seen_phi_ep) == 1, skill.seen_phi_ep
    assert len(comb.phi_ep_ids) == 1, comb.phi_ep_ids
    # (b) phi_step actually moved: at least one live aggregate changed across steps.
    uniq_live = set(live_seen)
    assert len(uniq_live) > 1, f"phi_step never changed across {steps} steps"
    # The skill saw many distinct phi_step identities (a fresh one per step).
    assert len(skill.seen_phi_step) > 1, skill.seen_phi_step
    # (c) kappa present + region-length on every step.
    R = len(phi_ep.region_centres)
    assert R > 0, R
    assert all(dl == R and sl == R for dl, sl in kappa_lens), (R, kappa_lens[:3])
    # (d) w reached the combiner.
    assert comb.saw_w, "combiner never received the objective w"

    print(f"[a] phi_ep static OK: 1 identity across {steps} steps, snapshot "
          f"unchanged (drivers={phi_ep.num_drivers}, scale={phi_ep.scale:.3f}).")
    print(f"[b] phi_step live OK: {len(uniq_live)} distinct live-aggregate tuples "
          f"over {steps} steps (skill saw {len(skill.seen_phi_step)} phi_step ids).")
    print(f"[c] kappa OK: region_demand/supply length == R == {R} on every step.")
    print(f"[d] w-routing OK: combiner saw w (skills structurally cannot -- their "
          f"signature has no w slot).")


def test_w_reaches_repositioner_not_skills() -> None:
    sc = _small_scenario()
    scorer = _ProbeReposScorer()
    repositioner = Repositioner(strength=1.0, scores_fn=scorer)
    skill = _ProbeSkill()
    comb = _ProbeCombiner()
    ctrl = DispatchController(comb, skills={"revenue": skill}, top_k=0,
                             repositioner=repositioner)
    r = rollout(build_env(sc), ctrl, _PREF, seed=0,
                reward_fn=_w, objective_label="served-minutes")
    # end-to-end finite metrics under the new signatures + an objective w.
    assert np.isfinite(r["service_rate"]), r
    assert np.isfinite(r["revenue"]), r
    # The repositioner scorer saw kappa + w (it is a w-reader, like the combiner).
    assert scorer.n_calls > 0, "reposition scorer never called (no idle drivers?)"
    assert scorer.saw_kappa, "reposition scorer never saw kappa"
    assert scorer.saw_w, "reposition scorer never saw w"
    print(f"[e] end-to-end OK: rollout finite (service={r['service_rate']:.3f}, "
          f"revenue={r['revenue']:.1f}); reposition scorer saw kappa+w over "
          f"{scorer.n_calls} calls.")


if __name__ == "__main__":
    test_phi_ep_static_and_phi_step_live()
    test_w_reaches_repositioner_not_skills()
    print("\nALL TWO-LAYER-STATS (phi_ep / phi_step / kappa / w) OFFLINE CHECKS PASSED")
