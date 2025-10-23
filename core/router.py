from __future__ import annotations

import math
from typing import Dict, Optional

from core.interfaces import Context
from core.registry import BehaviorRegistry


class SimpleRouter:
    """Keyword-driven router that scores behaviors based on metadata."""

    def __init__(self, registry: BehaviorRegistry, adapters: Optional[Dict[str, float]] = None):
        self.registry = registry
        self.adapters = adapters or {}

    def decide(self, ctx: Context) -> Dict[str, float]:
        text = (ctx.get("text") or ctx.get("data", {}).get("text") or "").lower()
        logits: Dict[str, float] = {}

        for name in self.registry.list():
            try:
                meta = self.registry.meta(name)
            except KeyError:
                meta = {}

            keywords = meta.get("keywords") or []
            hits = 0.0
            for keyword in keywords:
                if not isinstance(keyword, str):
                    continue
                hits += text.count(keyword.lower())

            bias = self.adapters.get(name, 0.0)
            logits[name] = 0.1 + hits + bias

        if not logits:
            return {}

        max_logit = max(logits.values())
        exps = {name: math.exp(value - max_logit) for name, value in logits.items()}
        total = sum(exps.values()) or 1.0
        return {name: exps[name] / total for name in logits}

    def choose(self, ctx: Context) -> str:
        scores = self.decide(ctx)
        if not scores:
            raise ValueError("No behaviors available for routing.")
        return max(scores.items(), key=lambda item: item[1])[0]
