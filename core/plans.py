from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List

import yaml

from core.interpreter import Interpreter
from core.rewards import aggregate_reward, ensure_reward_dict


def _apply_params(ctx: Dict[str, Any], params: Dict[str, Any]) -> None:
    for key, value in params.items():
        if key == "data" and isinstance(value, dict):
            ctx.setdefault("data", {}).update(value)
        else:
            ctx[key] = copy.deepcopy(value)


def run_plan(
    plan_path: str,
    ctx: Dict[str, Any],
    registry: Any,
    interpreter: Interpreter,
) -> Dict[str, Any]:
    path = Path(plan_path)
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    steps = spec.get("steps") or []
    _ = registry  # registry is available for future validation hooks

    print(f"Executing plan: {spec.get('name', path.stem)}")

    results: List[Dict[str, Any]] = []
    total_reward = 0.0

    for index, step in enumerate(steps, start=1):
        behavior = step.get("behavior")
        if not behavior:
            continue

        params = step.get("params") or {}
        if isinstance(params, dict):
            _apply_params(ctx, params)

        result = interpreter.execute(behavior, ctx)
        reward_dict = ensure_reward_dict(result.get("reward"))
        aggregate = aggregate_reward(reward_dict)
        total_reward += aggregate

        print(f"  Step {index}: {behavior} -> ok={result.get('ok')} reward={aggregate:.3f}")

        results.append({"behavior": behavior, "result": result, "reward": reward_dict})
        _record_plan_step(ctx, spec.get("name", path.stem), index, behavior, result, reward_dict)

    print(f"Plan total reward: {total_reward:.3f}")
    return {"steps": results, "total_reward": total_reward}


def _record_plan_step(
    ctx: Dict[str, Any],
    plan_name: str,
    step_index: int,
    behavior: str,
    result: Dict[str, Any],
    reward: Dict[str, float],
) -> None:
    if not isinstance(ctx, dict):
        return

    history = ctx.setdefault("plan_history", [])
    if isinstance(history, list):
        history.append(
            {
                "plan": plan_name,
                "step": step_index,
                "behavior": behavior,
                "reward": reward,
                "checks": result.get("checks") or {},
                "ok": bool(result.get("ok")),
            }
        )
