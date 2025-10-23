import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.audit import log_run
from core.features import extract_feature_key
from core.interpreter import Interpreter
from core.learn import BanditLearner
from core.plans import run_plan
from core.registry import BehaviorRegistry
from core.router import SimpleRouter
from core.rewards import aggregate_reward, ensure_reward_dict

BIAS_FILE = Path("adapter_biases.json")


def load_adapter_biases(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    result: Dict[str, Dict[str, float]] = {}
    if isinstance(data, dict):
        for feature_key, mapping in data.items():
            if isinstance(mapping, dict):
                result[str(feature_key)] = {
                    str(behavior): float(value)
                    for behavior, value in mapping.items()
                }
    return result


def save_adapter_biases(path: Path, biases: Dict[str, Dict[str, float]]) -> None:
    serializable = {
        feature_key: {behavior: float(value) for behavior, value in mapping.items()}
        for feature_key, mapping in biases.items()
    }
    path.write_text(json.dumps(serializable, indent=2, sort_keys=True), encoding="utf-8")


def _format_reward(value: Any) -> str:
    if isinstance(value, dict):
        try:
            return json.dumps({k: round(float(v), 3) for k, v in value.items()}, sort_keys=True)
        except (TypeError, ValueError):
            return json.dumps(value, sort_keys=True)
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "0.000"


def build_context(args: argparse.Namespace) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {
        "text": args.text,
        "data": {
            "text": args.text,
            "max_words": args.max_words,
        },
    }
    if args.dry_run:
        ctx["dry_run"] = True
    return ctx


def report_result(
    chosen: str,
    result: Dict[str, Any],
    ctx: Dict[str, Any],
    initial_data: Dict[str, Any],
    iteration: Optional[int] = None,
) -> None:
    if iteration is not None:
        print(f"\nIteration {iteration}")

    print(f"Behavior: {chosen}")
    print(f"Reward: {_format_reward(result.get('reward'))}")

    checks = result.get("checks") or {}
    if checks:
        print("Checks:")
        for name, score in checks.items():
            print(f"  {name}: {score:.3f}")

    outputs = {
        key: value
        for key, value in ctx["data"].items()
        if initial_data.get(key) != value or key not in initial_data
    }
    if outputs:
        print("Outputs:")
        print(json.dumps(outputs, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Super AI behaviors via CLI.")
    parser.add_argument("--text", required=True, help="Input text to process.")
    parser.add_argument(
        "--max_words",
        type=int,
        default=120,
        help="Maximum words allowed in the summary.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Flag behaviors to avoid side effects where supported.",
    )
    parser.add_argument(
        "--learn",
        type=int,
        default=0,
        help="Number of learning iterations to perform (0 for single run).",
    )
    parser.add_argument(
        "--plan",
        type=str,
        default="",
        help="Execute the behaviors defined in a YAML plan file.",
    )
    args = parser.parse_args()

    if args.plan and args.learn:
        parser.error("--plan cannot be combined with --learn.")

    registry = BehaviorRegistry().discover().load_meta()
    interpreter = Interpreter(registry)
    router = SimpleRouter(registry)

    initial_biases = load_adapter_biases(BIAS_FILE)
    learner = BanditLearner()
    for feature_key, mapping in initial_biases.items():
        learner.values[feature_key] = dict(mapping)

    if args.plan:
        ctx = build_context(args)
        plan_result = run_plan(args.plan, ctx, registry, interpreter)
        step_summaries = [
            {
                "behavior": entry["behavior"],
                "ok": bool(entry["result"].get("ok")),
                "reward": float(entry["result"].get("reward") or 0.0),
            }
            for entry in plan_result["steps"]
        ]
        log_run(
            {
                "mode": "plan",
                "plan": args.plan,
                "total_reward": plan_result["total_reward"],
                "steps": step_summaries,
                "biases": initial_biases,
            }
        )
        save_adapter_biases(BIAS_FILE, initial_biases)
        return

    ctx = build_context(args)
    ctx.setdefault("router", {})
    persisted_biases: Dict[str, Dict[str, float]] = {
        feature_key: dict(mapping) for feature_key, mapping in initial_biases.items()
    }

    iterations = args.learn if args.learn > 0 else 1
    iteration_records: List[Dict[str, Any]] = []

    for idx in range(iterations):
        iteration = idx + 1 if args.learn > 0 else None
        base_data = ctx.setdefault("data", {})
        base_data["text"] = args.text
        base_data["max_words"] = args.max_words
        ctx["text"] = args.text

        feature_key = extract_feature_key(ctx)
        router.adapters = dict(
            persisted_biases.get(feature_key, persisted_biases.get("default", {}))
        )

        initial_data = dict(ctx["data"])
        chosen = router.choose(ctx)
        result = interpreter.execute(chosen, ctx)
        reward_dict = ensure_reward_dict(result.get("reward"))
        result["reward"] = reward_dict

        report_result(chosen, result, ctx, initial_data, iteration)

        learner.update(chosen, reward_dict, feature_key)
        new_biases = learner.adapter_biases(feature_key)
        if new_biases:
            persisted_biases[feature_key] = new_biases
        elif feature_key in persisted_biases:
            persisted_biases.pop(feature_key)
        router.adapters = dict(new_biases)

        iteration_records.append(
            {
                "iteration": iteration or 1,
                "feature_key": feature_key,
                "behavior": chosen,
                "ok": bool(result.get("ok")),
                "reward": reward_dict,
            }
        )
        ctx.setdefault("plan_history", []).append(
            {
                "plan": "learn_loop" if args.learn > 0 else "single_run",
                "step": iteration or 1,
                "behavior": chosen,
                "reward": reward_dict,
                "checks": result.get("checks") or {},
                "ok": bool(result.get("ok")),
            }
        )

    if args.learn > 0:
        print("\nFinal adapter biases:")
        print(json.dumps(persisted_biases, indent=2))

    log_run(
        {
            "mode": "learn" if args.learn > 0 else "single",
            "iterations": iteration_records,
            "biases": persisted_biases,
        }
    )
    save_adapter_biases(BIAS_FILE, persisted_biases)


if __name__ == "__main__":
    main()
