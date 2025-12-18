from __future__ import annotations

import math
from typing import Any, Dict, Optional


class UCB1:
    def __init__(self) -> None:
        self.counts: Dict[str, int] = {}
        self.totals: Dict[str, float] = {}

    def ensure_arm(self, arm: str) -> None:
        if arm not in self.counts:
            self.counts[arm] = 0
            self.totals[arm] = 0.0

    def record(self, arm: str, reward: float) -> None:
        self.ensure_arm(arm)
        self.counts[arm] += 1
        self.totals[arm] += reward

    def pick(self) -> Optional[str]:
        if not self.counts:
            return None
        for arm, count in self.counts.items():
            if count == 0:
                return arm
        total_rounds = sum(self.counts.values())
        if total_rounds == 0:
            return next(iter(self.counts))
        best_arm = None
        best_score = float("-inf")
        for arm, count in self.counts.items():
            avg_reward = self.totals[arm] / count if count else 0.0
            bonus = math.sqrt(2.0 * math.log(total_rounds) / count)
            value = avg_reward + bonus
            if value > best_score:
                best_score = value
                best_arm = arm
        return best_arm

    def has_arms(self) -> bool:
        return bool(self.counts)

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return {
            "counts": self.counts,
            "totals": self.totals,
        }

    def load_state(self, payload: Dict[str, Any]) -> None:
        counts = payload.get("counts", {})
        totals = payload.get("totals", {})
        if isinstance(counts, dict):
            for arm, value in counts.items():
                try:
                    self.counts[arm] = int(value)
                except (TypeError, ValueError):
                    continue
        if isinstance(totals, dict):
            for arm, value in totals.items():
                try:
                    self.totals[arm] = float(value)
                except (TypeError, ValueError):
                    continue
        for arm in set(self.counts) | set(self.totals):
            self.counts.setdefault(arm, 0)
            self.totals.setdefault(arm, 0.0)


__all__ = ["UCB1"]
