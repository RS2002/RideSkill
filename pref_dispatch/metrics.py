"""Episode metrics: efficiency terms + driver-income fairness (proposal 5.4).

An :class:`EpisodeMetrics` accumulates per-step ``info["events"]`` and the
per-driver reward stream over a rollout, and reports:

* **efficiency** -- total revenue proxy, service rate, mean end-to-end service
  time, mean pickup / detour.
* **fairness**   -- driver-income **Gini** and coefficient of variation over the
  per-driver cumulative reward (the income the fairness budget equalises).

``income`` here is the running per-driver cumulative reward, which is also what
:class:`FairnessBudget` consumes online -- so the metric and the mechanism agree
on what "income" means.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


def gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative array. 0 = perfectly equal.

    Income can be negative in principle (penalties); we shift to non-negative
    before computing so the coefficient stays well-defined and comparable.
    """
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return 0.0
    x = x - min(x.min(), 0.0)  # shift so the minimum is >= 0
    if x.sum() == 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    index = np.arange(1, n + 1)
    return float((2.0 * (index * x).sum()) / (n * x.sum()) - (n + 1.0) / n)


@dataclass
class EpisodeMetrics:
    """Accumulates a single rollout's efficiency and fairness outcome."""

    income: Dict[int, float] = field(default_factory=dict)
    wage: Dict[int, float] = field(default_factory=dict)
    revenue: float = 0.0
    assigned: int = 0
    completed: int = 0
    total_orders: int = 0
    service_time_sum: float = 0.0
    service_time_n: int = 0
    detour_sum: float = 0.0
    # Reposition activity (Feature 3). ``empty_distance`` is all empty-car
    # movement; ``reposition_distance`` is the slice of it made while RELOCATING
    # (i.e. cruising to a demand region). Both stay 0 when reposition is off.
    empty_distance: float = 0.0
    reposition_distance: float = 0.0
    relocation_moves: int = 0

    def update(
        self,
        rewards: Dict[int, float],
        info: Dict,
        driver_status: Dict[int, str] = None,
    ) -> None:
        from pref_dispatch.wage import driver_wage_from_event

        for did, r in rewards.items():
            self.income[did] = self.income.get(did, 0.0) + float(r)

        events = info.get("events", {})
        for did, ev in events.items():
            # Per-driver take-home wage (fare, no penalties) -- what the
            # fairness budget equalizes; kept separate from reward income.
            self.wage[did] = self.wage.get(did, 0.0) + driver_wage_from_event(ev)
            for oid in ev.get("assigned_orders", []):
                self.assigned += 1
                party = ev.get("assigned_party_sizes", {}).get(oid, 1)
                self.revenue += ev.get("assigned_solo_times", {}).get(oid, 0.0) * party
                st = ev.get("assigned_service_times", {}).get(oid)
                if st is not None:
                    self.service_time_sum += st
                    self.service_time_n += 1
            self.completed += len(ev.get("completed_orders", []))
            self.detour_sum += ev.get("extra_detour_time", 0.0)
            # Reposition attribution: an empty move made while the driver's
            # post-step status is RELOCATING is a repositioning cruise.
            if ev.get("is_empty_move"):
                dm = float(ev.get("distance_moved", 0.0))
                self.empty_distance += dm
                if driver_status is not None and driver_status.get(did) == "relocating":
                    self.reposition_distance += dm
                    self.relocation_moves += 1

    def finalize(self, total_orders: int) -> Dict[str, float]:
        self.total_orders = total_orders
        incomes = np.array(list(self.income.values()), dtype=float)
        wages = np.array(list(self.wage.values()), dtype=float)
        service_rate = self.assigned / total_orders if total_orders else 0.0
        mean_service = (
            self.service_time_sum / self.service_time_n
            if self.service_time_n
            else 0.0
        )
        return {
            "revenue": self.revenue,
            "service_rate": service_rate,
            "completed": self.completed,
            "assigned": self.assigned,
            "mean_service_time": mean_service,
            "detour_total": self.detour_sum,
            "income_gini": gini(incomes),
            "income_cv": float(incomes.std() / (abs(incomes.mean()) + 1e-9)),
            "income_mean": float(incomes.mean()) if incomes.size else 0.0,
            "income_min": float(incomes.min()) if incomes.size else 0.0,
            # Driver-WAGE fairness (Feature 2): Gini over take-home fare.
            "wage_gini": gini(wages),
            "wage_cv": float(wages.std() / (abs(wages.mean()) + 1e-9)),
            "wage_mean": float(wages.mean()) if wages.size else 0.0,
            "wage_min": float(wages.min()) if wages.size else 0.0,
            # Reposition activity (Feature 3): fraction of empty-car distance
            # spent cruising to demand, and the raw counts. 0 when reposition off.
            "relocation_moves": self.relocation_moves,
            "reposition_distance_km": self.reposition_distance,
            "reposition_distance_ratio": (
                self.reposition_distance / self.empty_distance
                if self.empty_distance > 0
                else 0.0
            ),
        }
