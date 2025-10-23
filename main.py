import argparse
import copy
import json
from typing import Any, Dict, Optional

from core.interpreter import Interpreter
from core.learn import BanditLearner
from core.registry import BehaviorRegistry
from core.router import SimpleRouter


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
    print(f"Reward: {result.get('reward', 0.0):.3f}")

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
    args = parser.parse_args()

    registry = BehaviorRegistry().discover().load_meta()
    router = SimpleRouter(registry)
    interpreter = Interpreter(registry)
    learner = BanditLearner() if args.learn > 0 else None

    base_ctx = build_context(args)
    iterations = args.learn if args.learn > 0 else 1

    for idx in range(iterations):
        iteration = idx + 1 if args.learn > 0 else None
        ctx = copy.deepcopy(base_ctx)
        initial_data = dict(ctx["data"])

        chosen = router.choose(ctx)
        result = interpreter.execute(chosen, ctx)

        report_result(chosen, result, ctx, initial_data, iteration)

        if learner is not None:
            reward = float(result.get("reward") or 0.0)
            learner.update(chosen, reward)
            router.adapters = learner.adapter_biases()

    if learner is not None:
        print("\nFinal adapter biases:")
        print(json.dumps(router.adapters, indent=2))


if __name__ == "__main__":
    main()
