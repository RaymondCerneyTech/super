from __future__ import annotations

import math
from typing import Dict, Optional


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


__all__ = ["UCB1"]
