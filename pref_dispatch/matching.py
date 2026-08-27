"""Per-driver softmax(+no-op) x budget, then greedy one-to-one matching.

This is the inner mechanism of proposal 4.2, steps 2-4:

1. For each driver ``d`` and each *feasible* pending order ``o``, blend the
   combiner-selected skills' scores into ``a[d, o]``; also compute the no-op
   score ``a[d, NOOP]``.
2. Softmax over ``d``'s own candidate set ``{orders} + {NOOP}`` so the row sums
   to 1 (cross-driver / cross-skill comparability; the no-op competes for mass
   so a driver that should wait can).
3. Multiply the softmax weights by the fairness budget ``beta_d``.
4. Greedy **one-to-one** matching on ``beta_d * softmax`` : each order to at most
   one driver, each driver at most one new order this step. This yields a
   conflict-free bid set the env accepts (the env raises on double-bid orders).

"One-to-one per step" is the ride-*sharing* formulation from the proposal: an
en-route driver can still be matched a new order, and pooling emerges over
multiple steps.

Two-layer stats (final-version redesign): skills and the combiner now take
``(phi_ep, phi_step)`` instead of a single ``phi`` plus ``dist`` -- the travel-time
closure lives on ``phi_ep.dist``. The combiner additionally reads the episode
objective ``w`` (carried on ``phi_ep.reward_fn``); skills never do.

Every-pair scoring (A3): the default ``top_k`` is a LARGE cap (:data:`DEFAULT_TOP_K`
= 60), so at realistic operating points every feasible nearby order is scored; the
KNN prune remains only as the safety valve that keeps 1000-car rollouts O(N*K).
``top_k <= 0`` scores every pair (true all-pairs, for small envs / parity checks).

Top-k skill blending (v6 item 8). Two skills' raw scores are NOT on a common
scale: a revenue skill returns dollars (tens), a service skill returns a
normalized 0..1 quantity. The old blend was a plain weighted sum of those raw
numbers, so a 50/50 weight dict was in practice ~98% revenue -- the combiner
could not actually mix. Now each driver's candidate set is scored per skill and
each skill's column is standardized ACROSS THAT DRIVER'S CANDIDATES before the
weighted sum, so a weight really is a share of the decision. Only the
``blend_k`` highest-weighted skills participate (``blend_k <= 0`` = all).
``blend_k = 1`` is exactly the old one-hot path: a single skill's raw scores go
into the softmax untouched, so nothing is rescaled and behaviour is identical.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from pref_dispatch.combiner import Combiner
from pref_dispatch.global_stats import EpisodeStats, GlobalStats
from pref_dispatch.preference import Preference
from pref_dispatch.skills import Skill

Dist = Callable[[tuple, tuple], float]
RewardFn = Callable[[Dict], float]
NOOP = -1  # sentinel "order id" for the no-op option.

# Large default candidate cap: at the evaluation operating points every feasible
# nearby order is scored (A3 "score every feasible pair"); pruning only engages as
# a safety valve when a driver has more than this many pending pickups nearby.
DEFAULT_TOP_K = 60

# How many of the combiner's skills actually take part in one driver's decision.
# 1 = the old one-hot behaviour (exactly reproduced); >1 blends that many after
# per-skill standardization; <=0 blends every positively-weighted skill.
#
# 2026-08-10: raised 1 -> 3. It shipped at 1, which reproduced the pre-v6 one-hot
# path exactly -- i.e. the knob existed but was switched off, so v6 item 8 bought
# nothing. Measured consequence at 1: the champion combiner had only TWO distinct
# fleet behaviours across five reward families (pooling vs everything else),
# because one driver-step could express only ONE skill out of eight. 3 of 8 lets a
# driver mix a primary with two supports while still being interpretable ("this
# car is mostly fast-serve with some coverage"). Training and evaluation both read
# this constant, so they cannot drift apart.
DEFAULT_BLEND_K = 3


def _blended_score(
    skills: Dict[str, Skill],
    weights: Dict[str, float],
    driver_obs: Dict,
    order: Optional[Dict],
    phi_ep: EpisodeStats,
    phi_step: GlobalStats,
    noop: bool = False,
) -> float:
    """Weighted sum of the named skills' (order- or no-op) RAW scores.

    Kept for single-pair probes and parity checks. The live matcher uses
    :func:`_candidate_scores`, which standardizes each skill across the driver's
    whole candidate set first (raw scores are not on a common scale).
    """
    total = 0.0
    wsum = 0.0
    for name, w in weights.items():
        if w == 0.0 or name not in skills:
            continue
        sk = skills[name]
        val = (
            sk.noop_score(driver_obs, phi_ep, phi_step)
            if noop
            else sk.score(driver_obs, order, phi_ep, phi_step)
        )
        total += w * val
        wsum += w
    return total / wsum if wsum > 0 else 0.0


def _active_skills(
    skills: Dict[str, Skill],
    weights: Dict[str, float],
    blend_k: int,
) -> List[Tuple[str, float]]:
    """The ``blend_k`` highest-weighted known skills, weights renormalized to 1.

    Ties are broken by name so the selection is deterministic. ``blend_k <= 0``
    keeps every positively-weighted known skill.
    """
    items = [
        (name, float(w))
        for name, w in weights.items()
        if name in skills and float(w) > 0.0
    ]
    if not items:
        return []
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    if blend_k > 0:
        items = items[:blend_k]
    tot = sum(w for _, w in items)
    if tot <= 0.0:
        return []
    return [(name, w / tot) for name, w in items]


def _candidate_scores(
    skills: Dict[str, Skill],
    active: List[Tuple[str, float]],
    driver_obs: Dict,
    orders: List[Dict],
    phi_ep: EpisodeStats,
    phi_step: GlobalStats,
    feasible_floor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Blended score of every candidate for one driver.

    Returns ``(scores, feasible)``, both of length ``len(orders) + 1``; the last
    entry is the no-op (always feasible, it is the driver's fallback option).

    An order is infeasible as soon as ANY participating skill floors it -- a
    capacity violation is a hard fact about the (driver, order) pair, not
    something a second skill's high score should be able to outvote (the old
    weighted-sum test could let a 0.01-weighted floor slip through).

    With a single active skill the raw scores are returned untouched, so the
    softmax downstream sees exactly what the pre-v6 one-hot path fed it. With
    two or more, each skill's column is z-scored over the feasible candidates
    first: skills answer "which of these orders" on a shared scale, and the
    weights become real shares. A skill that is flat across the candidate set
    (zero spread) contributes nothing, which is the correct reading of "this
    skill has no opinion here".
    """
    n = len(orders) + 1
    m = len(active)
    raw = np.empty((n, m), dtype=float)
    for j, (name, _) in enumerate(active):
        sk = skills[name]
        for i, order in enumerate(orders):
            raw[i, j] = sk.score(driver_obs, order, phi_ep, phi_step)
        raw[n - 1, j] = sk.noop_score(driver_obs, phi_ep, phi_step)

    feasible = np.ones(n, dtype=bool)
    if len(orders):
        feasible[: n - 1] = (raw[: n - 1, :] > feasible_floor).all(axis=1)

    if m == 1:
        return raw[:, 0], feasible

    z = np.zeros_like(raw)
    for j in range(m):
        col = raw[feasible, j]
        sd = float(col.std())
        if sd > 1e-12:
            z[:, j] = (raw[:, j] - float(col.mean())) / sd
    w_vec = np.array([w for _, w in active], dtype=float)
    return z @ w_vec, feasible


def _softmax(vals: np.ndarray, temperature: float) -> np.ndarray:
    z = vals / max(temperature, 1e-6)
    z = z - z.max()  # numerical stability
    e = np.exp(z)
    return e / e.sum()


def _knn_candidates(
    observations: Dict[int, Dict],
    pending: List[Dict],
    top_k: int,
) -> Dict[int, List[int]]:
    """For each driver, the indices (into ``pending``) of its K nearest orders.

    A city-scale fleet must NOT score every (driver, order) pair -- that is the
    O(N_drivers x N_pending) blow-up that makes rollouts unusable at 1000 cars.
    Real dispatch systems only consider spatially nearby orders, so we pre-filter
    to each driver's K closest pickups using a single vectorised distance
    computation (straight-line proxy; the skills still do the exact routing-aware
    scoring on the survivors). This both bounds cost at O(N_drivers x K) and is
    more realistic (a car across the city was never a real candidate).

    ``top_k <= 0`` disables pruning (score all orders) -- used for small envs and
    for checking the pruning has no qualitative effect.
    """
    n_pending = len(pending)
    if top_k <= 0 or n_pending <= top_k:
        allidx = list(range(n_pending))
        return {did: allidx for did in observations}

    # Order pickup coordinates, once.
    ox = np.fromiter((o["origin"][0] for o in pending), dtype=float, count=n_pending)
    oy = np.fromiter((o["origin"][1] for o in pending), dtype=float, count=n_pending)

    out: Dict[int, List[int]] = {}
    for did, obs in observations.items():
        dx, dy = obs["self"]["location"]
        d2 = (ox - dx) ** 2 + (oy - dy) ** 2
        # argpartition is O(n): the K smallest distances, order within irrelevant
        # (the skill scoring re-ranks them anyway).
        out[did] = np.argpartition(d2, top_k)[:top_k].tolist()
    return out


def compute_bids(
    observations: Dict[int, Dict],
    skills: Dict[str, Skill],
    combiner: Combiner,
    phi_ep: EpisodeStats,
    phi_step: GlobalStats,
    budgets: Dict[int, float],
    *,
    w: Optional[RewardFn] = None,
    temperature: float = 1.0,
    feasible_floor: float = -1e8,
    top_k: int = DEFAULT_TOP_K,
    blend_k: int = DEFAULT_BLEND_K,
) -> Tuple[Dict[int, List[int]], Dict[int, str]]:
    """Return conflict-free ``{driver_id: [order_id]}`` bids and driver classes.

    Only *feasible* (finite-score) orders enter a driver's softmax set, and only
    the driver's ``top_k`` spatially-nearest pending orders are scored at all
    (see :func:`_knn_candidates`; the default cap is large so every feasible pair
    is scored at realistic operating points). A driver whose winning option is the
    no-op (or who has no feasible nearby order) bids nothing. Each order is awarded
    to at most one driver (greedy by weight), guaranteeing no conflict.

    ``blend_k`` caps how many of the combiner's skills score one driver's
    candidates (see :func:`_candidate_scores`); ``1`` is the pre-v6 one-hot path.

    ``w`` is the episode objective (callable reward fn) forwarded to the combiner;
    skills never receive it.
    """
    any_obs = next(iter(observations.values()))
    pending = list(any_obs["pending_orders"])
    classes: Dict[int, str] = {}

    # Spatial pre-filter: each driver only considers its K nearest pickups.
    near = _knn_candidates(observations, pending, top_k)

    # Two quantities per (driver, order):
    #   * ``prefers_bid`` : does the driver *intrinsically* want this order over
    #     waiting? Decided on the UNBUDGETED softmax (order prob > no-op prob),
    #     so the fairness budget cannot force a driver to take an order it should
    #     refuse (protects the no-op semantics).
    #   * ``match_weight`` : the greedy-matching priority, which DOES include the
    #     budget, so among drivers that all want an order the low-income one wins.
    match_weight: Dict[Tuple[int, int], float] = {}
    prefers_bid: Dict[Tuple[int, int], bool] = {}

    for did, obs in observations.items():
        weights = combiner.weights_for(obs, phi_ep, phi_step, w)
        classes[did] = combiner.classify(obs, phi_ep, phi_step)

        active = _active_skills(skills, weights, blend_k)
        orders = [pending[oi] for oi in near[did]]
        if active:
            scores, feasible = _candidate_scores(
                skills, active, obs, orders, phi_ep, phi_step, feasible_floor
            )
        else:  # combiner named no known skill: the driver can only wait
            scores = np.zeros(len(orders) + 1, dtype=float)
            feasible = np.zeros(len(orders) + 1, dtype=bool)
            feasible[-1] = True

        cand_ids: List[int] = [
            orders[i]["order_id"] for i in range(len(orders)) if feasible[i]
        ]
        cand_scores: List[float] = [
            float(scores[i]) for i in range(len(orders)) if feasible[i]
        ]
        cand_ids.append(NOOP)
        cand_scores.append(float(scores[-1]))

        probs = _softmax(np.array(cand_scores, dtype=float), temperature)
        prob_map = {oid: p for oid, p in zip(cand_ids, probs)}
        noop_p = prob_map[NOOP]
        beta = budgets.get(did, 1.0)
        for oid, p in prob_map.items():
            if oid == NOOP:
                continue
            prefers_bid[(did, oid)] = p > noop_p  # unbudgeted intrinsic decision
            match_weight[(did, oid)] = beta * p   # budgeted matching priority

    # Greedy one-to-one matching, highest budgeted weight first. An edge is only
    # eligible if the driver intrinsically prefers bidding to waiting.
    order_taken: set = set()
    driver_taken: set = set()
    bids: Dict[int, List[int]] = {did: [] for did in observations}

    for (did, oid), w_edge in sorted(match_weight.items(), key=lambda kv: -kv[1]):
        if did in driver_taken or oid in order_taken:
            continue
        if not prefers_bid[(did, oid)]:
            continue  # driver would rather wait than take this order
        bids[did] = [oid]
        driver_taken.add(did)
        order_taken.add(oid)

    return bids, classes
