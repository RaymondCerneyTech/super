# core/learn.py
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional

FeatureValues = Dict[str, float]


class BanditLearner:
    """Exponential moving-average bandit learner that produces contextual adapter biases."""

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = max(0.0, min(1.0, alpha))
        self.values: Dict[str, FeatureValues] = defaultdict(dict)

    def update(
        self,
        chosen: str,
        reward: float,
        feature_key: str,
        alpha: Optional[float] = None,
    ) -> None:
        """Update the smoothed reward estimate for the chosen behavior and feature bucket."""
        bucket = self.values.setdefault(feature_key, {})
        a = self.alpha if alpha is None else max(0.0, min(1.0, alpha))
        current = bucket.get(chosen, 0.0)
        bucket[chosen] = (1 - a) * current + a * reward

    def adapter_biases(self, feature_key: str) -> Dict[str, float]:
        """Map EMAs within a feature bucket to mean-centered biases clamped to [-0.3, +0.3]."""
        if feature_key in self.values and self.values[feature_key]:
            bucket = self.values[feature_key]
        elif feature_key != "default" and "default" in self.values and self.values["default"]:
            bucket = self.values["default"]
        else:
            return {}

        vals = list(bucket.values())
        if len(vals) <= 1:
            mu = 0.0
        else:
            mu = sum(vals) / len(vals)
        tau = 0.5
        raw = {k: (v - mu) / (tau + 1e-8) for k, v in bucket.items()}
        return {k: max(-0.3, min(0.3, x)) for k, x in raw.items()}
