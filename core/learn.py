# core/learn.py
from typing import Dict
from collections import defaultdict

class BanditLearner:
    def __init__(self):
        self.counts = defaultdict(int)
        self.values = defaultdict(float)  # running mean reward per behavior

    def update(self, chosen: str, reward: float):
        n = self.counts[chosen] = self.counts[chosen] + 1
        v = self.values[chosen]
        self.values[chosen] = v + (reward - v) / n

    def adapter_biases(self) -> Dict[str, float]:
        # normalize to small biases (-0.5..+0.5)
        if not self.values: return {}
        mn, mx = min(self.values.values()), max(self.values.values())
        rng = (mx - mn) or 1.0
        return {k: (v - mn)/rng - 0.5 for k,v in self.values.items()}
