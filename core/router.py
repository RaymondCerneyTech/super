from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

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
        ranking = sorted(
            self.decide(ctx).items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if not ranking:
            raise ValueError("No behaviors available for routing.")

        primary = ranking[0][0]
        should_fallback, reason = self._should_fallback(primary, ctx)
        if should_fallback and len(ranking) > 1:
            fallback = ranking[1][0]
            self._log_fallback(ctx, primary, fallback, reason)
            return fallback
        return primary

    def _should_fallback(self, behavior: str, ctx: Context) -> Tuple[bool, Optional[str]]:
        reward, checks = self._extract_last_result(behavior, ctx)
        if reward is not None and reward <= 0.0:
            return True, f"reward={reward}"
        if checks is not None:
            if len(checks) == 0:
                return True, "no checks produced"

        try:
            meta = self.registry.meta(behavior)
        except KeyError:
            meta = {}
        success_checks = meta.get("success_checks") or []
        if not success_checks:
            return True, "no success checks configured"
        return False, None

    def _extract_last_result(
        self, behavior: str, ctx: Context
    ) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
        if not isinstance(ctx, dict):
            return None, None

        router_state = ctx.get("router")
        if not isinstance(router_state, dict):
            router_state = {}

        recent_results = router_state.get("recent_results")
        if isinstance(recent_results, dict):
            result = recent_results.get(behavior)
            if isinstance(result, dict):
                reward = result.get("reward")
                checks = result.get("checks")
                reward_value = None
                if reward is not None:
                    try:
                        reward_value = float(reward)
                    except (TypeError, ValueError):
                        reward_value = None
                if isinstance(checks, dict):
                    return reward_value, checks
                return reward_value, None

        data = ctx.get("data")
        reward_value = None
        checks_value: Optional[Dict[str, Any]] = None
        if isinstance(data, dict):
            reward_map = data.get("rewards")
            if isinstance(reward_map, dict) and behavior in reward_map:
                try:
                    reward_value = float(reward_map[behavior])
                except (TypeError, ValueError):
                    reward_value = None

            checks_map = data.get("checks")
            if isinstance(checks_map, dict):
                candidate = checks_map.get(behavior)
                if isinstance(candidate, dict):
                    checks_value = candidate

        return reward_value, checks_value

    def _log_fallback(self, ctx: Context, fallback_from: str, fallback_to: str, reason: Optional[str]) -> None:
        if not isinstance(ctx, dict):
            return

        audit_log = ctx.setdefault("router_audit", [])
        if isinstance(audit_log, list):
            audit_log.append(
                {
                    "fallback_from": fallback_from,
                    "fallback_to": fallback_to,
                    "reason": reason,
                }
            )

        message = f"[router] fallback {fallback_from} -> {fallback_to}"
        if reason:
            message += f" ({reason})"
        print(message)
