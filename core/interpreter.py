# core/interpreter.py
from typing import Any, Dict, List, Tuple

from core.checks import CHECKS
from core.interfaces import Context, Result
from core.registry import BehaviorRegistry

class Interpreter:
    def __init__(self, registry: BehaviorRegistry):
        self.registry = registry

    def execute(self, name: str, ctx: Context) -> Result:
        behavior = self.registry.get(name)
        data = ctx.setdefault("data", {})

        # preconditions
        for key in behavior.inputs:
            if key not in data and key not in ctx:
                return {
                    "ok": False,
                    "logs": [f"Missing input: {key}"],
                    "checks": {},
                    "reward": 0.0,
                }

        result = behavior.run(ctx)

        # write outputs back into ctx.data
        output = result.get("output") or {}
        for key, value in output.items():
            data[key] = value

        checks, reward = self._evaluate_checks(name, ctx)

        final_result: Result = dict(result)
        final_result["checks"] = checks
        final_result["reward"] = reward
        final_result["output"] = output
        return final_result

    def _evaluate_checks(self, name: str, ctx: Context) -> Tuple[Dict[str, float], float]:
        try:
            meta = self.registry.meta(name)
        except KeyError:
            return {}, 0.0

        success_checks = meta.get("success_checks") or []
        if not isinstance(success_checks, list):
            return {}, 0.0

        scores: Dict[str, float] = {}
        reward = 0.0

        for entry in success_checks:
            if not isinstance(entry, dict):
                continue
            check_type = entry.get("type")
            if check_type not in CHECKS:
                continue

            fn = CHECKS[check_type]
            weight = float(entry.get("weight", 1.0))
            args = self._build_check_args(check_type, entry, ctx)

            if any(arg is None for arg in args):
                score = 0.0
            else:
                try:
                    score = float(fn(*args))
                except Exception:
                    score = 0.0

            threshold = entry.get("threshold")
            if threshold is not None:
                score = 1.0 if score >= float(threshold) else 0.0

            score = max(0.0, min(1.0, score))
            scores[check_type] = score
            reward += weight * score

        return scores, reward

    def _build_check_args(self, check_type: str, entry: Dict[str, Any], ctx: Context) -> List[Any]:
        if check_type == "length_leq":
            return [
                self._resolve_value(entry.get("target"), ctx),
                self._resolve_value(entry.get("arg"), ctx),
            ]
        if check_type == "similarity_cosine":
            return [
                self._resolve_value(entry.get("source"), ctx),
                self._resolve_value(entry.get("target"), ctx),
            ]
        if check_type == "tone_keyword_match":
            tone_value = self._resolve_value(entry.get("arg"), ctx)
            if tone_value is None:
                tone_value = entry.get("tone")
            return [
                self._resolve_value(entry.get("target"), ctx),
                tone_value,
            ]

        args = entry.get("args")
        if isinstance(args, list):
            return [self._resolve_value(arg, ctx) for arg in args]
        return []

    def _resolve_value(self, key: Any, ctx: Context) -> Any:
        if key is None or not isinstance(key, (str, int, float)):
            return key

        data = ctx.get("data") or {}
        if isinstance(key, str):
            if key in data:
                return data[key]
            return ctx.get(key)

        # numeric literal
        return key
