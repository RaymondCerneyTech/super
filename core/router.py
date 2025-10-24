from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Tuple

from core.interfaces import Context
from core.registry import BehaviorRegistry
from core.rewards import ensure_reward_dict


class SimpleRouter:
    """Keyword-driven router with simple cluster biasing."""

    ANALYTIC_KEYWORDS = {"summary", "summarize", "format", "policy", "report", "exact", "concise"}
    CREATIVE_KEYWORDS = {"creative", "story", "rewrite", "tone", "style"}

    def __init__(self, registry: BehaviorRegistry, adapters: Optional[Dict[str, float]] = None) -> None:
        self.registry = registry
        self.adapters = adapters or {}
        self.cluster_values: Dict[str, float] = {"analytic": 0.6, "creative": 0.6}
        self.cluster_counts: Dict[str, int] = {"analytic": 1, "creative": 1}

    # ------------------------------------------------------------------
    # Public API
    def cluster_hint(self, goal_text: str, ctx: Optional[Context] = None) -> str:
        goal_text = (goal_text or "").lower()
        analytic_score = self.cluster_values.get("analytic", 0.5)
        creative_score = self.cluster_values.get("creative", 0.5)

        for keyword in self.ANALYTIC_KEYWORDS:
            if keyword in goal_text:
                analytic_score += 0.1
        for keyword in self.CREATIVE_KEYWORDS:
            if keyword in goal_text:
                creative_score += 0.1

        if ctx:
            text = (ctx.get("text") or ctx.get("data", {}).get("text") or "").lower()
            if len(text.split()) > 80:
                analytic_score += 0.05

        return "creative" if creative_score > analytic_score else "analytic"

    def register_outcome(self, cluster: str, rewards: Dict[str, float]) -> None:
        overall = ensure_reward_dict(rewards).get("overall", 0.0)
        prev = self.cluster_values.get(cluster, 0.5)
        alpha = 0.2
        self.cluster_values[cluster] = prev + alpha * (overall - prev)
        self.cluster_counts[cluster] = self.cluster_counts.get(cluster, 0) + 1

    def register_bandit_outcome(self, ctx: Context, rewards: Dict[str, float]) -> None:
        if not isinstance(ctx, dict):
            return
        router_state = ctx.setdefault("router", {})
        if not isinstance(router_state, dict):
            return
        reward_dict = ensure_reward_dict(rewards)
        router_state.setdefault("recent_results", {})
        router_state["recent_results"].setdefault("_last", {"reward": 0.0})
        router_state["recent_results"]["_last"] = {
            "reward": reward_dict.get("overall", 0.0),
            "rewards": reward_dict,
            "checks": ctx.get("data", {}).get("checks", {}),
        }

    def decide(self, ctx: Context) -> Dict[str, float]:
        text = (ctx.get("text") or ctx.get("data", {}).get("text") or "").lower()
        router_state = ctx.get("router") if isinstance(ctx, dict) else {}
        goal_text = ""
        if isinstance(router_state, dict):
            goal_text = str(router_state.get("goal_text") or "")
        cluster_bias = router_state.get("cluster_bias") if isinstance(router_state, dict) else None
        if not cluster_bias:
            cluster_bias = self.cluster_hint(goal_text or text, ctx if isinstance(ctx, dict) else None)
            if isinstance(router_state, dict):
                router_state["cluster_bias"] = cluster_bias

        logits: Dict[str, float] = {}

        for name in self.registry.list():
            try:
                meta = self.registry.meta(name)
            except KeyError:
                meta = {}

            keywords = meta.get("keywords") or []
            success_checks = meta.get("success_checks") or []
            capabilities = [cap.lower() for cap in meta.get("capabilities", [])]

            score = 0.1
            for keyword in keywords:
                if not isinstance(keyword, str):
                    continue
                if keyword.lower() in text:
                    score += 0.5
            if keywords:
                score += 0.2
            else:
                score -= 0.2

            if not success_checks:
                score -= 0.3

            if cluster_bias == "analytic" and "creative" in capabilities:
                score -= 0.2
            if cluster_bias == "creative" and "creative" in capabilities:
                score += 0.2

            score += self.adapters.get(name, 0.0)

            logits[name] = score

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

    # ------------------------------------------------------------------
    # Internal helpers
    def _should_fallback(self, behavior: str, ctx: Context) -> Tuple[bool, Optional[str]]:
        reward, checks = self._extract_last_result(behavior, ctx)
        if reward is not None and reward <= 0.0:
            return True, f"reward={reward:.3f}"
        if checks is not None and len(checks) == 0:
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
            result = recent_results.get(behavior) or recent_results.get("_last")
            if isinstance(result, dict):
                reward = result.get("reward")
                checks = result.get("checks") if isinstance(result.get("checks"), dict) else None
                reward_value = None
                if reward is not None:
                    try:
                        reward_value = float(reward)
                    except (TypeError, ValueError):
                        reward_value = None
                return reward_value, checks

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


__all__ = ["SimpleRouter"]
