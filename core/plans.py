from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List

import yaml

from core.interpreter import Interpreter


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
        reward = float(result.get("reward") or 0.0)
        total_reward += reward

        print(f"  Step {index}: {behavior} -> ok={result.get('ok')} reward={reward:.3f}")

        results.append({"behavior": behavior, "result": result})

    print(f"Plan total reward: {total_reward:.3f}")
    return {"steps": results, "total_reward": total_reward}
