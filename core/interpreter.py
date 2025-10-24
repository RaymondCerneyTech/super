# core/interpreter.py
from typing import Any, Dict, List, Optional, Tuple

from core.checks import CHECKS
from core.interfaces import Context, Result
from core.metrics import compute_reward_factors
from core.rewards import (
    aggregate_reward,
    apply_explanation_bonus,
    compute_explanation_scores,
    ensure_reward_dict,
    merge_rewards,
)
from core.registry import BehaviorRegistry


class Interpreter:
    def __init__(self, registry: BehaviorRegistry):
        self.registry = registry

    def execute(self, name: str, ctx: Context) -> Result:
        behavior = self.registry.get(name)
        data = ctx.setdefault("data", {})

        for key in behavior.inputs:
            if key not in data and key not in ctx:
                return {
                    "ok": False,
                    "logs": [f"Missing input: {key}"],
                    "checks": {},
                    "reward": 0.0,
                    "rewards": {"overall": 0.0},
                    "rationale": {"why": "precondition failed", "evidence": []},
                    "effects": [],
                }

        meta = self._get_meta(name)
        validation_errors = self._validate_args(meta, ctx)
        if validation_errors:
            return {
                "ok": False,
                "logs": validation_errors,
                "checks": {},
                "reward": 0.0,
                "rewards": {"overall": 0.0},
                "rationale": {"why": "argument validation failed", "evidence": validation_errors},
                "effects": [],
            }

        result = behavior.run(ctx) or {}

        output = result.get("output") or {}
        for key, value in output.items():
            data[key] = value

        checks, base_reward = self._evaluate_checks(name, ctx, meta)
        reward_dict = ensure_reward_dict(base_reward)
        reward_keys = [key for key in reward_dict.keys() if key != "overall"]

        ok = bool(result.get("ok", True))

        if ok:
            metrics_bonus = compute_reward_factors(name, ctx, output)
            if metrics_bonus:
                merge_rewards(reward_dict, metrics_bonus.items())
            behavior_rewards = result.get("reward") or result.get("rewards")
            if behavior_rewards:
                merge_rewards(reward_dict, ensure_reward_dict(behavior_rewards).items())
        else:
            reward_dict = {key: 0.0 for key in reward_keys}
            reward_dict["overall"] = 0.0

        rationale = result.get("rationale") if "rationale" in result else None
        if not rationale:
            rationale = self._default_rationale(name, meta)
        explanation_scores = compute_explanation_scores(rationale, output) if ok else {"explanation_presence": 0.0, "explanation_specificity": 0.0, "explanation_alignment": 0.0}
        reward_dict = apply_explanation_bonus(reward_dict, explanation_scores) if ok else reward_dict

        effects = result["effects"] if "effects" in result else self.registry.behavior_effects(name)

        final_result: Result = dict(result)
        final_result["checks"] = checks
        final_result["output"] = output
        final_result["rationale"] = rationale
        final_result["effects"] = effects
        final_result["rewards"] = reward_dict
        final_result["reward"] = reward_dict

        ok = bool(result.get("ok", True))
        self._store_run_outcome(ctx, name, reward_dict, checks, output, rationale, effects, ok)
        return final_result

    def _evaluate_checks(
        self, name: str, ctx: Context, meta: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        if meta is None:
            try:
                meta = self.registry.meta(name)
            except KeyError:
                return {}, {"overall": 0.0}

        success_checks = meta.get("success_checks") or []
        if not isinstance(success_checks, list):
            return {}, {"overall": 0.0}

        scores: Dict[str, float] = {}
        reward_components: Dict[str, float] = {}
        weighted_sum = 0.0
        weight_total = 0.0

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

            component_key = str(entry.get("reward_key") or check_type)
            reward_components[component_key] = score
            weighted_sum += weight * score
            weight_total += weight

        reward_components["overall"] = weighted_sum / weight_total if weight_total else (
            sum(reward_components.values()) / len(reward_components) if reward_components else 0.0
        )

        return scores, reward_components

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

        if expected == "list":
            if not isinstance(value, list):
                return value, f"Invalid type for {name}: expected list."
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

        return key

    def _store_run_outcome(
        self,
        ctx: Context,
        behavior: str,
        reward: Dict[str, float],
        checks: Dict[str, float],
        output: Dict[str, Any],
        rationale: Dict[str, Any],
        effects: List[str],
        ok: bool,
    ) -> None:
        if not isinstance(ctx, dict):
            return

        reward_dict = ensure_reward_dict(reward)
        if not ok:
            reward_dict["overall"] = 0.0

        router_state = ctx.setdefault("router", {})
        recent = None
        entry = {
            "reward": reward_dict,
            "rewards": reward_dict,
            "checks": checks,
            "ok": ok,
            "effects": effects,
            "rationale": rationale,
            "_behavior": behavior,
        }
        if isinstance(router_state, dict):
            recent = router_state.setdefault("recent_results", {})
            if isinstance(recent, dict):
                recent[behavior] = entry

        data = ctx.setdefault("data", {})
        if isinstance(data, dict):
            rewards_map = data.setdefault("rewards", {})
            if isinstance(rewards_map, dict):
                rewards_map[behavior] = reward_dict.copy()

            checks_map = data.setdefault("checks", {})
            if isinstance(checks_map, dict):
                checks_map[behavior] = checks

        backlog = ctx.setdefault("reward_backlog", {})
        if not isinstance(backlog, dict):
            return

        if recent is None:
            recent = router_state.setdefault("recent_results", {})

        for factor, value in reward_dict.items():
            if factor == "overall":
                continue
            if value >= 1.0:
                pending = backlog.get(factor, [])
                resolved = []
                for prev_entry in list(pending):
                    if not isinstance(prev_entry, dict):
                        continue
                    prev_behavior = prev_entry.get("_behavior", behavior)
                    prev_reward = ensure_reward_dict(prev_entry.get("reward"))
                    merge_rewards(prev_reward, [(factor, (prev_reward.get(factor, 0.0) + value) / 2)])
                    prev_entry["reward"] = prev_reward
                    if isinstance(data, dict):
                        data.setdefault("rewards", {})[prev_behavior] = prev_reward.copy()
                    resolved.append(prev_entry)
                if pending:
                    backlog[factor] = [item for item in pending if item not in resolved]
                    if not backlog[factor]:
                        backlog.pop(factor, None)
            else:
                bucket = backlog.setdefault(factor, [])
                if entry not in bucket:
                    bucket.append(entry)

    def _default_rationale(self, behavior: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        why = meta.get("description") or f"Executed {behavior}"
        return {"why": str(why), "evidence": []}


__all__ = ["Interpreter"]
