import argparse
import json
import sys
from textwrap import dedent

from core.interpreter import Interpreter
from core.planner import plan_task
from core.registry import BehaviorRegistry
from core.rewards import ensure_reward_dict


def build_registry() -> BehaviorRegistry:
    return BehaviorRegistry().discover().load_meta()


def build_interpreter() -> Interpreter:
    return Interpreter(build_registry())


def print_reward(result: dict) -> None:
    reward = ensure_reward_dict(result.get("reward"))
    print("Reward:")
    print(json.dumps(reward, indent=2))


def command_help(args: argparse.Namespace) -> None:
    help_text = dedent(
        """
        Available commands:

          help
              Show this overview with command descriptions and examples.

          summarize --text TEXT [--strategy {trim,extractive,abstractive}] [--max-words N] [--max-sentences N]
              Generate a summary. Example:
                python main.py summarize --text "Long report text..." --strategy extractive --max-words 60

          format --text TEXT [--style {business,casual}] [--wrap-width N]
              Normalize spacing, headings, and bullets. Example:
                python main.py format --text "# heading..." --style business

          optimize --text TEXT [--platform P] [--topic T] [--max-length N]
              Optimise a social media post and suggest hashtags. Example:
                python main.py optimize --text "Announcing our launch" --platform twitter --topic marketing

          check --text TEXT --policies "phrase1,phrase2"
              Scan content for prohibited phrases and redact them. Example:
                python main.py check --text "Share secret roadmap" --policies "secret roadmap"

          plan --goal "goal description" [--text TEXT] [--execute]
              Build a behaviour plan for a goal, optionally execute it.
        """
    ).strip()
    print(help_text)


def command_summarize(args: argparse.Namespace) -> None:
    interpreter = build_interpreter()
    ctx = {
        "text": args.text,
        "data": {
            "text": args.text,
            "max_words": args.max_words,
            "max_sentences": args.max_sentences,
            "summary_strategy": args.strategy,
        },
    }
    result = interpreter.execute("summarize", ctx)
    summary = ctx["data"].get("summary", "")
    print("Summary:\n" + summary)
    print_reward(result)


def command_format(args: argparse.Namespace) -> None:
    interpreter = build_interpreter()
    ctx = {
        "text": args.text,
        "data": {
            "text": args.text,
            "format_style": args.style,
            "wrap_width": args.wrap_width,
        },
    }
    result = interpreter.execute("document_formatting", ctx)
    formatted = ctx["data"].get("formatted_text", "")
    print("Formatted Document:\n" + formatted)
    print_reward(result)


def command_optimize(args: argparse.Namespace) -> None:
    interpreter = build_interpreter()
    ctx = {
        "text": args.text,
        "data": {
            "text": args.text,
            "platform": args.platform,
            "topic": args.topic,
            "max_length": args.max_length,
        },
    }
    result = interpreter.execute("social_post_optimize", ctx)
    print("Optimized Text:\n" + ctx["data"].get("optimized_text", ""))
    print("Hashtags:", ", ".join(ctx["data"].get("hashtags", [])))
    print_reward(result)


def command_check(args: argparse.Namespace) -> None:
    interpreter = build_interpreter()
    policies = [item.strip() for item in args.policies.split(",") if item.strip()]
    ctx = {
        "text": args.text,
        "data": {
            "text": args.text,
            "policies": policies,
            "policy_replacement": args.replacement,
        },
    }
    result = interpreter.execute("policy_check", ctx)
    print("Sanitized Text:\n" + ctx["data"].get("sanitized_text", ""))
    print("Violations:")
    for violation in ctx["data"].get("violations", []):
        print(f" - {violation['phrase']} -> {violation['context']}")
    print_reward(result)


def command_plan(args: argparse.Namespace) -> None:
    registry = build_registry()
    sequence = plan_task(
        goal=args.goal,
        registry=registry,
        learner=None,
        history=None,
        max_depth=args.max_depth,
    )
    if not sequence:
        print("No plan could be generated for that goal.")
        return
    print("Plan:", " -> ".join(sequence))
    if not args.execute:
        return

    if not args.text:
        print("Execution requires --text to seed the context.", file=sys.stderr)
        sys.exit(1)

    interpreter = Interpreter(registry)
    ctx = {"text": args.text, "data": {"text": args.text}}
    for behavior in sequence:
        print(f"\nRunning {behavior}...")
        result = interpreter.execute(behavior, ctx)
        print_reward(result)
    print("\nFinal context data:")
    print(json.dumps(ctx["data"], indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Modular behavior CLI")
    subparsers = parser.add_subparsers(dest="command")

    help_parser = subparsers.add_parser("help", help="Show command overview")
    help_parser.set_defaults(func=command_help)

    summarize_parser = subparsers.add_parser("summarize", help="Generate a summary")
    summarize_parser.add_argument("--text", required=True, help="Text to summarize")
    summarize_parser.add_argument("--strategy", choices=["trim", "extractive", "abstractive"], default="extractive")
    summarize_parser.add_argument("--max-words", type=int, default=120)
    summarize_parser.add_argument("--max-sentences", type=int, default=5)
    summarize_parser.set_defaults(func=command_summarize)

    format_parser = subparsers.add_parser("format", help="Format a document")
    format_parser.add_argument("--text", required=True, help="Document text")
    format_parser.add_argument("--style", choices=["business", "casual"], default="business")
    format_parser.add_argument("--wrap-width", type=int, default=80)
    format_parser.set_defaults(func=command_format)

    optimize_parser = subparsers.add_parser("optimize", help="Optimize a social media post")
    optimize_parser.add_argument("--text", required=True)
    optimize_parser.add_argument("--platform", choices=["twitter", "linkedin", "instagram"], default="twitter")
    optimize_parser.add_argument("--topic", default="business")
    optimize_parser.add_argument("--max-length", type=int, default=280)
    optimize_parser.set_defaults(func=command_optimize)

    check_parser = subparsers.add_parser("check", help="Perform policy compliance checks")
    check_parser.add_argument("--text", required=True)
    check_parser.add_argument("--policies", required=True, help="Comma separated list")
    check_parser.add_argument("--replacement", default="[REDACTED]")
    check_parser.set_defaults(func=command_check)

    plan_parser = subparsers.add_parser("plan", help="Generate a behavior plan")
    plan_parser.add_argument("--goal", required=True)
    plan_parser.add_argument("--max-depth", type=int, default=3)
    plan_parser.add_argument("--text", help="Optional text seed for execution")
    plan_parser.add_argument("--execute", action="store_true")
    plan_parser.set_defaults(func=command_plan)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        command_help(args)
        return 0
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
