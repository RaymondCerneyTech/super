# core/learn.py
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict


class BanditLearner:
    """Simple running-mean bandit learner that produces small adapter biases."""

    def __init__(self) -> None:
        self.counts = defaultdict(int)
        self.values = defaultdict(float)  # running mean reward per behavior

    def update(self, chosen: str, reward: float) -> None:
        """Update the running mean reward for the chosen behavior."""
        n = self.counts[chosen] = self.counts[chosen] + 1
        v = self.values[chosen]
        self.values[chosen] = v + (reward - v) / n

    def adapter_biases(self) -> Dict[str, float]:
        """Map running means to adapter biases in [-0.5, +0.5]."""
        if not self.values:
            return {}

        means = self.values
        mn, mx = min(means.values()), max(means.values())

        if math.isclose(mx, mn):
            if mx > 0:
                return {k: 0.5 for k in means}
            if mx < 0:
                return {k: -0.5 for k in means}
            return {k: 0.0 for k in means}

        rng = mx - mn
        return {k: ((v - mn) / rng) - 0.5 for k, v in means.items()}
