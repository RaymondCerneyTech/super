from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.interfaces import Context, evaluate_preconditions
from core.registry import BehaviorRegistry
from core.rewards import aggregate_reward, ensure_reward_dict


def extract_features(goal_text: str, text: str, data: Optional[Dict[str, Any]] = None) -> List[float]:
    goal = (goal_text or "").lower()
    content = text or ""
    content_lower = content.lower()
    combined = f"{goal} {content_lower}".strip()

    length_norm = min(1.0, len(content) / 500.0) if content else 0.0
    concise = 1.0 if any(token in combined for token in ("concise", "summary", "brief", "short")) else 0.0
    compliant = 1.0 if any(token in combined for token in ("compliant", "compliance", "policy", "redact")) else 0.0
    formatted = 1.0 if any(token in combined for token in ("formatted", "format", "structure", "layout")) else 0.0
    creative = 1.0 if any(token in combined for token in ("creative", "story", "rewrite", "narrative")) else 0.0
    tone = 1.0 if "tone" in combined else 0.0
    exact = 1.0 if any(token in combined for token in ("exact", "verbatim", "precise", "literal")) else 0.0
    wants_verbose = 1.0 if any(token in combined for token in ("verbose", "max", "long-form", "extended")) else 0.0
    wants_cited = 1.0 if "cited" in combined or "citation" in combined else 0.0
    wants_grounded = 1.0 if "grounded" in combined or "ground truth" in combined else 0.0
    wants_fresh = 1.0 if "fresh" in combined or "recent" in combined else 0.0

    k_passages = 12
    max_chars = 12000
    if isinstance(data, dict):
        k_passages = int(data.get("k_passages") or k_passages)
        max_chars = int(data.get("max_chars") or max_chars)
        if data.get("min_words") and int(data.get("min_words")) > 800:
            wants_verbose = 1.0
        if data.get("fresh_days"):
            wants_fresh = 1.0

    k_norm = min(1.0, k_passages / 20.0)
    chars_norm = min(1.0, max_chars / 20000.0)

    features = [
        1.0,
        float(length_norm),
        concise,
        compliant,
        formatted,
        creative,
        tone,
        exact,
        wants_verbose,
        wants_cited,
        wants_grounded,
        wants_fresh,
        k_norm,
        chars_norm,
    ]
    return [float(value) for value in features]


class LinUCBDisjoint:
    def __init__(self, arms: List[str], dimension: int, alpha: float = 0.3) -> None:
        self.arms = list(arms)
        self.dimension = dimension
        self.alpha = alpha
        self.A_inv: Dict[str, List[List[float]]] = {arm: self._identity(dimension) for arm in self.arms}
        self.b: Dict[str, List[float]] = {arm: [0.0] * dimension for arm in self.arms}

    def select(self, context: List[float]) -> str:
        best_arm = self.arms[0]
        best_score = float("-inf")
        for arm in self.arms:
            theta = self._mat_vec(self.A_inv[arm], self.b[arm])
            mean = self._dot(theta, context)
            variance = self._confidence(self.A_inv[arm], context)
            score = mean + self.alpha * variance
            if score > best_score + 1e-12:
                best_score = score
                best_arm = arm
        return best_arm

    def update(self, arm: str, context: List[float], reward: float) -> None:
        if arm not in self.A_inv:
            return
        A_inv = self.A_inv[arm]
        x = context
        A_inv_x = self._mat_vec(A_inv, x)
        denom = 1.0 + self._dot(x, A_inv_x)
        if denom <= 1e-9:
            denom = 1e-9
        factor = 1.0 / denom
        adjustment = self._matrix_scalar(self._outer(A_inv_x, A_inv_x), factor)
        self.A_inv[arm] = self._matrix_sub(A_inv, adjustment)
        self.b[arm] = [b_i + reward * x_i for b_i, x_i in zip(self.b[arm], x)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arms": list(self.arms),
            "dimension": self.dimension,
            "alpha": self.alpha,
            "A_inv": {arm: matrix for arm, matrix in self.A_inv.items()},
            "b": {arm: vector for arm, vector in self.b.items()},
        }

    def load_state(self, payload: Dict[str, Any]) -> None:
        arms = payload.get("arms") or self.arms
        dimension = int(payload.get("dimension", self.dimension))
        if dimension != self.dimension or set(arms) != set(self.arms):
            return
        self.alpha = float(payload.get("alpha", self.alpha))
        matrices = payload.get("A_inv", {})
        vectors = payload.get("b", {})
        for arm in self.arms:
            matrix = matrices.get(arm)
            if isinstance(matrix, list):
                self.A_inv[arm] = [list(map(float, row)) for row in matrix]
            vector = vectors.get(arm)
            if isinstance(vector, list):
                self.b[arm] = list(map(float, vector))

    @staticmethod
    def _identity(dimension: int) -> List[List[float]]:
        return [[1.0 if i == j else 0.0 for j in range(dimension)] for i in range(dimension)]

    @staticmethod
    def _dot(a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    @staticmethod
    def _mat_vec(matrix: List[List[float]], vector: List[float]) -> List[float]:
        return [sum(row[i] * vector[i] for i in range(len(vector))) for row in matrix]

    @staticmethod
    def _outer(a: List[float], b: List[float]) -> List[List[float]]:
        return [[x * y for y in b] for x in a]

    @staticmethod
    def _matrix_scalar(matrix: List[List[float]], scalar: float) -> List[List[float]]:
        return [[value * scalar for value in row] for row in matrix]

    @staticmethod
    def _matrix_sub(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
        return [[value - b[row_index][col_index] for col_index, value in enumerate(row)] for row_index, row in enumerate(a)]

    def _confidence(self, A_inv: List[List[float]], context: List[float]) -> float:
        A_inv_x = self._mat_vec(A_inv, context)
        variance = self._dot(context, A_inv_x)
        return math.sqrt(max(variance, 0.0))


class SimpleRouter:
    """Keyword-driven router with a light contextual bandit."""

    def __init__(self, registry: BehaviorRegistry, adapters: Optional[Dict[str, float]] = None):
        self.registry = registry
        self.adapters = adapters or {}
        self.cluster_values: Dict[str, float] = {"analytic": 0.6, "creative": 0.6}
        self.cluster_counts: Dict[str, int] = {"analytic": 1, "creative": 1}
        feature_dim = len(extract_features("", "", {}))
        self._bandit = LinUCBDisjoint(["analytic", "creative"], feature_dim)
        self._state_path = self._resolve_state_path()
        if self._state_path:
            self._load_state()

    def cluster_hint(self, goal_text: str, ctx: Optional[Context] = None) -> str:
        router_state: Dict[str, Any] = {}
        if isinstance(ctx, dict):
            router_state = ctx.setdefault("router", {})
            existing = router_state.get("cluster_bias")
            if isinstance(existing, str) and existing:
                return existing
        goal_text = goal_text or ""
        text = ""
        if isinstance(ctx, dict):
            text = (ctx.get("text") or ctx.get("data", {}).get("text") or "")
        data = {}
        if isinstance(ctx, dict):
            data = ctx.get("data", {}) if isinstance(ctx.get("data"), dict) else {}
        features = extract_features(goal_text, text, data)
        if isinstance(router_state, dict):
            router_state["bandit_features"] = features
            router_state["bandit_disabled"] = bool(router_state.get("no_bandit", False))
        use_bandit = isinstance(router_state, dict) and not router_state.get("no_bandit", False)
        if use_bandit:
            arm = self._bandit.select(features)
            router_state["bandit_arm"] = arm
            router_state["cluster_bias"] = arm
            return arm
        heuristic = self._heuristic_cluster(goal_text, ctx)
        if isinstance(router_state, dict):
            router_state["bandit_arm"] = heuristic
            router_state["cluster_bias"] = heuristic
        return heuristic

    def register_outcome(self, cluster: str, rewards: Dict[str, float]) -> None:
        overall = ensure_reward_dict(rewards).get("overall", 0.0)
        prev = self.cluster_values.get(cluster, 0.5)
        alpha = 0.2
        self.cluster_values[cluster] = prev + alpha * (overall - prev)
        self.cluster_counts[cluster] = self.cluster_counts.get(cluster, 0) + 1
        self._persist_state()

    def register_bandit_outcome(self, ctx: Context, rewards: Dict[str, float]) -> None:
        if not isinstance(ctx, dict):
            return
        router_state = ctx.get("router")
        if not isinstance(router_state, dict):
            return
        if router_state.get("no_bandit"):
            return
        arm = router_state.get("bandit_arm")
        features = router_state.get("bandit_features")
        if not isinstance(arm, str) or not isinstance(features, list):
            return
        reward_value = ensure_reward_dict(rewards).get("overall", 0.0)
        try:
            feature_vector = [float(value) for value in features]
        except (TypeError, ValueError):
            return
        self._bandit.update(arm, feature_vector, reward_value)
        self._persist_state()

    def decide(self, ctx: Context) -> Dict[str, float]:
        text = (ctx.get("text") or ctx.get("data", {}).get("text") or "").lower()
        router_state = ctx.get("router") if isinstance(ctx, dict) else {}
        bias = None
        if isinstance(router_state, dict):
            bias = router_state.get("cluster_bias")
            text = (router_state.get("goal_text") or text).lower()
        if not bias:
            bias = self.cluster_hint(text, ctx)

        logits: Dict[str, float] = {}

        current_flags: List[str] = []
        if isinstance(ctx, dict):
            flag_value = ctx.get("flags")
            if isinstance(flag_value, (list, set, tuple)):
                current_flags = list(flag_value)

        for name in self.registry.list():
            try:
                meta = self.registry.meta(name)
            except KeyError:
                meta = {}

            preconditions = meta.get("preconditions", ["true"])
            try:
                allowed = evaluate_preconditions(preconditions, ctx, current_flags)
            except Exception:
                allowed = True
            if not allowed:
                continue

            keywords = meta.get("keywords") or []
            hits = 0.0
            for keyword in keywords:
                if not isinstance(keyword, str):
                    continue
                hits += text.count(keyword.lower())

            bias_adjustment = self._capability_bias(name, bias)
            hits += bias_adjustment

            bias = bias or "analytic"
            bias_hits = self.adapters.get(name, 0.0)
            logits[name] = 0.1 + hits + bias_hits

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

    def _heuristic_cluster(self, goal_text: str, ctx: Optional[Context] = None) -> str:
        goal_lower = (goal_text or "").lower()
        analytic_score = self.cluster_values.get("analytic", 0.6)
        creative_score = self.cluster_values.get("creative", 0.6)

        analytic_goal_keywords = {"summary", "compliant", "formatted", "exact", "concise"}
        creative_goal_keywords = {"creative", "tone", "style", "rewrite"}

        if any(keyword in goal_lower for keyword in analytic_goal_keywords):
            return "analytic"
        if any(keyword in goal_lower for keyword in creative_goal_keywords):
            return "creative"

        analytic_keywords = ["concise", "compliant", "summary", "report", "format", "exact"]
        creative_keywords = ["creative", "tone", "story", "social", "rewrite", "brand"]

        for kw in analytic_keywords:
            if kw in goal_lower:
                analytic_score += 0.1
        for kw in creative_keywords:
            if kw in goal_lower:
                creative_score += 0.1

        if ctx:
            text = (ctx.get("text") or ctx.get("data", {}).get("text") or "")
            if len(text.split()) > 80:
                analytic_score += 0.05

        return "creative" if creative_score > analytic_score else "analytic"

    def _capability_bias(self, behavior: str, cluster_bias: Optional[str]) -> float:
        if not cluster_bias:
            return 0.0
        caps = [cap.lower() for cap in self.registry.behavior_capabilities(behavior)]
        if cluster_bias == "analytic":
            return -0.2 if "creative" in caps else 0.1
        if cluster_bias == "creative":
            return 0.2 if "creative" in caps else -0.05
        return 0.0

    def _should_fallback(self, behavior: str, ctx: Context) -> Tuple[bool, Optional[str]]:
        reward, checks, ok_flag = self._extract_last_result(behavior, ctx)
        if ok_flag is False:
            return True, "previous failure"
        reward_value = self._aggregate_reward(reward)
        if reward_value is not None and reward_value <= 0.0:
            return True, f"reward={reward_value:.3f}"
        if checks is not None:
            if len(checks) == 0:
                return True, "no checks produced"

        try:
            meta = self.registry.meta(behavior)
        except KeyError:
            meta = {}
        success_checks = meta.get("success_checks") or []
        if not success_checks:
            produced_effects = meta.get("effects") or []
            if produced_effects:
                return False, None
            return True, "no success checks configured"
        return False, None

    def _extract_last_result(
        self, behavior: str, ctx: Context
    ) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, Any]], Optional[bool]]:
        if not isinstance(ctx, dict):
            return None, None, None

        router_state = ctx.get("router")
        if not isinstance(router_state, dict):
            router_state = {}

        recent_results = router_state.get("recent_results")
        if isinstance(recent_results, dict):
            result = recent_results.get(behavior) or recent_results.get("_last")
            if isinstance(result, dict):
                reward = ensure_reward_dict(result.get("reward"))
                checks = result.get("checks") if isinstance(result.get("checks"), dict) else None
                ok_flag = result.get("ok") if isinstance(result.get("ok"), bool) else None
                return reward, checks, ok_flag

        data = ctx.get("data")
        reward_value: Optional[Dict[str, float]] = None
        checks_value: Optional[Dict[str, Any]] = None
        if isinstance(data, dict):
            reward_map = data.get("rewards")
            if isinstance(reward_map, dict) and behavior in reward_map:
                reward_value = ensure_reward_dict(reward_map[behavior])

            checks_map = data.get("checks")
            if isinstance(checks_map, dict):
                candidate = checks_map.get(behavior)
                if isinstance(candidate, dict):
                    checks_value = candidate

        return reward_value, checks_value, None

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

    def _aggregate_reward(self, reward: Optional[Dict[str, float]]) -> Optional[float]:
        if reward is None:
            return None
        return aggregate_reward(reward)

    def _resolve_state_path(self) -> Optional[Path]:
        if os.getenv("SUPER_ROUTER_STATE_DISABLE"):
            return None
        custom_path = os.getenv("SUPER_ROUTER_STATE_PATH")
        if custom_path:
            return Path(custom_path)
        directory_env = os.getenv("SUPER_ROUTER_STATE_DIR")
        base_dir = Path(directory_env) if directory_env else Path("data") / "bandit"
        return base_dir / "router_state.json"

    def _load_state(self) -> None:
        if not self._state_path or not self._state_path.exists():
            return
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return

        cluster_values = payload.get("cluster_values")
        if isinstance(cluster_values, dict):
            for key, value in cluster_values.items():
                try:
                    self.cluster_values[key] = float(value)
                except (TypeError, ValueError):
                    continue

        cluster_counts = payload.get("cluster_counts")
        if isinstance(cluster_counts, dict):
            for key, value in cluster_counts.items():
                try:
                    self.cluster_counts[key] = int(value)
                except (TypeError, ValueError):
                    continue

        bandit_state = payload.get("bandit")
        if isinstance(bandit_state, dict):
            self._bandit.load_state(bandit_state)

    def _persist_state(self) -> None:
        if not self._state_path:
            return
        payload = {
            "cluster_values": self.cluster_values,
            "cluster_counts": self.cluster_counts,
            "bandit": self._bandit.to_dict(),
        }
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            with self._state_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle)
        except OSError:
            return


__all__ = ["SimpleRouter", "LinUCBDisjoint", "extract_features"]
