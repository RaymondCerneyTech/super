# core/interpreter.py
from typing import Any, Dict, List, Optional, Tuple

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

        meta = self._get_meta(name)
        validation_errors = self._validate_args(meta, ctx)
        if validation_errors:
            return {
                "ok": False,
                "logs": validation_errors,
                "checks": {},
                "reward": 0.0,
            }

        result = behavior.run(ctx)

        # write outputs back into ctx.data
        output = result.get("output") or {}
        for key, value in output.items():
            data[key] = value

        checks, reward = self._evaluate_checks(name, ctx, meta)
        self._store_run_outcome(ctx, name, reward, checks)

        final_result: Result = dict(result)
        final_result["checks"] = checks
        final_result["reward"] = reward
        final_result["output"] = output
        return final_result

    def _evaluate_checks(
        self, name: str, ctx: Context, meta: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, float], float]:
        if meta is None:
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

    def _get_meta(self, name: str) -> Dict[str, Any]:
        try:
            meta = self.registry.meta(name)
            if isinstance(meta, dict):
                return meta
        except KeyError:
            pass
        return {}

    def _validate_args(self, meta: Dict[str, Any], ctx: Context) -> List[str]:
        args_spec = meta.get("args") or {}
        if not isinstance(args_spec, dict):
            return []

        errors: List[str] = []
        data = ctx.setdefault("data", {})

        for arg_name, spec in args_spec.items():
            if not isinstance(spec, dict):
                continue

            value_present = False
            if arg_name in data:
                value = data[arg_name]
                value_present = True
            elif arg_name in ctx:
                value = ctx[arg_name]
                data[arg_name] = value
                value_present = True
            elif "default" in spec:
                value = spec["default"]
                data[arg_name] = value
                value_present = True
            else:
                value = None

            if not value_present:
                if spec.get("required"):
                    errors.append(f"Missing required argument: {arg_name}")
                continue

            value = data.get(arg_name)
            coerced, type_error = self._coerce_type(arg_name, value, spec.get("type"))
            if type_error:
                errors.append(type_error)
                continue

            canonical, enum_error = self._validate_enum(arg_name, coerced, spec.get("enum"))
            if enum_error:
                errors.append(enum_error)
                continue

            data[arg_name] = canonical
            if arg_name in ctx:
                ctx[arg_name] = canonical

        return errors

    def _coerce_type(self, name: str, value: Any, expected: Optional[str]) -> Tuple[Any, Optional[str]]:
        if expected is None or value is None:
            return value, None

        if expected == "str":
            if not isinstance(value, str):
                return str(value), None
            return value, None

        if expected == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                return value, f"Invalid type for {name}: expected int."
            return value, None

        if expected == "float":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return value, f"Invalid type for {name}: expected float."
            return float(value), None

        if expected == "bool":
            if not isinstance(value, bool):
                return value, f"Invalid type for {name}: expected bool."
            return value, None

        return value, None

    def _validate_enum(
        self, name: str, value: Any, enum_values: Optional[List[Any]]
    ) -> Tuple[Any, Optional[str]]:
        if not enum_values or value is None:
            return value, None
        if not isinstance(enum_values, list):
            return value, None

        normalized_map: Dict[Any, Any] = {}
        for option in enum_values:
            key = option.lower() if isinstance(option, str) else option
            normalized_map[key] = option

        comparison = value.lower() if isinstance(value, str) else value

        if comparison not in normalized_map:
            return value, f"Invalid value for {name}: {value}. Expected one of {enum_values}."

        canonical = normalized_map[comparison]
        return canonical, None

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

    def _store_run_outcome(self, ctx: Context, behavior: str, reward: float, checks: Dict[str, float]) -> None:
        if not isinstance(ctx, dict):
            return

        router_state = ctx.setdefault("router", {})
        if isinstance(router_state, dict):
            recent = router_state.setdefault("recent_results", {})
            if isinstance(recent, dict):
                recent[behavior] = {
                    "reward": reward,
                    "checks": checks,
                }

        data = ctx.setdefault("data", {})
        if isinstance(data, dict):
            rewards_map = data.setdefault("rewards", {})
            if isinstance(rewards_map, dict):
                rewards_map[behavior] = reward

            checks_map = data.setdefault("checks", {})
            if isinstance(checks_map, dict):
                checks_map[behavior] = checks
