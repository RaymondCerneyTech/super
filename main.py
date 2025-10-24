import argparse
import json
from datetime import datetime, timezone
import sys
from textwrap import dedent
from typing import Dict, List

from core.config import load_config
from core.interfaces import Context
from core.interpreter import Interpreter
from core.planner import normalise_goal_flags, plan
from core.registry import BehaviorRegistry
from core.rewards import ensure_reward_dict
from core.router import SimpleRouter
from core.logs import append_jsonl

_HELP_TEXT = dedent(
    """
    Available commands:

      help
          Show this overview with command descriptions and examples.

      summarize --text TEXT [--strategy {trim,extractive,abstractive}] [--max-words N] [--max-sentences N]
          Generate a summary. Example:
            python main.py summarize --text \"Long report text...\" --strategy extractive --max-words 60

      format --text TEXT [--style {business,casual}] [--wrap-width N]
          Normalize spacing, headings, and bullets. Example:
            python main.py format --text \"# heading...\" --style business

      optimize --text TEXT [--platform P] [--topic T] [--max-length N]
          Optimise a social media post and suggest hashtags. Example:
            python main.py optimize --text \"Announcing our launch\" --platform twitter --topic marketing

      check --text TEXT --policies \"phrase1,phrase2\"
          Scan content for prohibited phrases and redact them. Example:
            python main.py check --text \"Share secret roadmap\" --policies \"secret roadmap\"

      ingest --url URL | --rss FEED | --path FILE [--tags \"t1,t2\"] [--index-backend {tfidf,hnsw}] [--limit N]
          Fetch a URL or feed into the local index. Example:
            python main.py ingest --url \"https://example.com/guide\" --tags \"guide,example\" --index-backend hnsw

      ask --question \"...\" [--k-passages N] [--max-chars N] [--cited] [--grounded] [--verbose | --max-verbose | --min-words N] [--fresh DAYS] [--index-backend BACKEND] [--log-file PATH] [--no-bandit] [--explain]
          Retrieve knowledge and produce a cited, verbose answer. Example:
            python main.py ask --question \"Explain transformers\" --cited --grounded --verbose --min-words 1200 --explain

      plan --goal \"summary,compliant,formatted\" [--text TEXT] [--policies \"phrase1,...\"] [--log-file PATH] [--no-bandit] [--explain] [--max-expansions N]
          Construct a behaviour plan to satisfy goal flags. Example:
            python main.py plan --goal \"summary,compliant,formatted\" --text \"Draft...\" --log-file logs/bandit.jsonl --explain
    """
).strip()


def build_registry() -> BehaviorRegistry:
    return BehaviorRegistry().discover().load_meta()


def build_runtime() -> tuple[BehaviorRegistry, Interpreter, SimpleRouter]:
    registry = build_registry()
    interpreter = Interpreter(registry)
    router = SimpleRouter(registry)
    return registry, interpreter, router


def print_reward(result: dict) -> None:
    reward = ensure_reward_dict(result.get("rewards") or result.get("reward"))
    print("Reward:")
    print(json.dumps(reward, indent=2))


def command_help(args: argparse.Namespace) -> None:
    print(_HELP_TEXT)


def _prepare_context(
    text: str,
    goal_text: str,
    router: SimpleRouter,
    *,
    no_bandit: bool = False,
    log_file: str | None = None,
) -> Context:
    ctx: Context = {
        "text": text,
        "data": {"text": text},
        "router": {"goal_text": goal_text, "no_bandit": no_bandit},
    }
    if log_file:
        ctx["router"]["log_file"] = log_file
    cluster = router.cluster_hint(goal_text, ctx)
    ctx["router"]["cluster_bias"] = cluster
    return ctx


def _config_section(args: argparse.Namespace, section: str) -> Dict[str, object]:
    config = getattr(args, "config_data", None) or {}
    value = config.get(section, {})
    return value if isinstance(value, dict) else {}


def command_summarize(args: argparse.Namespace) -> None:
    registry, interpreter, router = build_runtime()
    ctx = _prepare_context(args.text, args.text, router)
    cluster = ctx["router"]["cluster_bias"]
    ctx["data"].update(
        {
            "max_words": args.max_words,
            "max_sentences": args.max_sentences,
            "summary_strategy": args.strategy,
        }
    )
    result = interpreter.execute("summarize", ctx)
    router.register_outcome(cluster, result.get("rewards", {}))
    summary = ctx["data"].get("summary", "")
    print("Summary:\n" + summary)
    print_reward(result)


def command_format(args: argparse.Namespace) -> None:
    registry, interpreter, router = build_runtime()
    ctx = _prepare_context(args.text, args.text, router)
    cluster = ctx["router"]["cluster_bias"]
    ctx["data"].update({"format_style": args.style, "wrap_width": args.wrap_width})
    result = interpreter.execute("document_formatting", ctx)
    router.register_outcome(cluster, result.get("rewards", {}))
    formatted = ctx["data"].get("formatted_text", "")
    print("Formatted Document:\n" + formatted)
    print_reward(result)


def command_optimize(args: argparse.Namespace) -> None:
    registry, interpreter, router = build_runtime()
    goal_text = f"optimize {args.platform} {args.topic}"
    ctx = _prepare_context(args.text, goal_text, router)
    cluster = ctx["router"]["cluster_bias"]
    ctx["data"].update({"platform": args.platform, "topic": args.topic, "max_length": args.max_length})
    result = interpreter.execute("social_post_optimize", ctx)
    router.register_outcome(cluster, result.get("rewards", {}))
    print("Optimized Text:\n" + ctx["data"].get("optimized_text", ""))
    print("Hashtags:", ", ".join(ctx["data"].get("hashtags", [])))
    print_reward(result)


def command_check(args: argparse.Namespace) -> None:
    registry, interpreter, router = build_runtime()
    policies = [item.strip() for item in args.policies.split(",") if item.strip()]
    goal_text = "compliance check"
    ctx = _prepare_context(args.text, goal_text, router)
    cluster = ctx["router"]["cluster_bias"]
    ctx["data"].update({"policies": policies, "policy_replacement": args.replacement})
    result = interpreter.execute("policy_check", ctx)
    router.register_outcome(cluster, result.get("rewards", {}))
    print("Sanitized Text:\n" + ctx["data"].get("sanitized_text", ""))
    print("Violations:")
    for violation in ctx["data"].get("violations", []):
        print(f" - {violation['phrase']} -> {violation['context']}")
    print_reward(result)


def command_ingest(args: argparse.Namespace) -> None:
    registry, interpreter, _ = build_runtime()
    config = _config_section(args, "ingest")
    index_backend = args.index_backend or config.get("index_backend", "tfidf")
    tags = args.tags if args.tags is not None else config.get("tags", "")
    limit = args.limit if args.limit is not None else config.get("limit", 15)
    lang_any = bool(args.lang_any or config.get("lang_any", False))

    data: Dict[str, object] = {
        "index_backend": index_backend,
        "tags": tags,
        "limit": limit,
    }
    if args.url:
        data["url"] = args.url
    if args.rss:
        data["rss"] = args.rss
    if args.path:
        data["path"] = args.path
    if lang_any:
        data["lang_any"] = True

    ctx: Context = {"data": data}
    result = interpreter.execute("ingest_web", ctx)
    ingested = ctx["data"].get("ingest_log", [])
    print(f"Docs ingested: {len(ingested)} (backend={index_backend})")
    skipped = result.get("output", {}).get("skipped", [])
    if skipped:
        print("Skipped items:")
        for item in skipped:
            print(f" - {item}")
    print_reward(result)


def command_ask(args: argparse.Namespace) -> None:
    registry, interpreter, router = build_runtime()
    config = _config_section(args, "ask")

    index_backend = args.index_backend or config.get("index_backend", "tfidf")
    k_passages = args.k_passages if args.k_passages is not None else config.get("k_passages", 12)
    max_chars = args.max_chars if args.max_chars is not None else config.get("max_chars", 12000)
    tags = args.tags if args.tags is not None else config.get("tags", "")
    fresh_days = args.fresh if args.fresh is not None else config.get("fresh_days")
    log_file = args.log_file or config.get("log_file")
    max_expansions = args.max_expansions if args.max_expansions is not None else config.get("max_expansions", 25)
    explain = bool(args.explain or config.get("explain", False))
    no_bandit = bool(args.no_bandit or config.get("no_bandit", False))

    if args.max_verbose:
        verbosity = "max"
    elif args.verbose:
        verbosity = "verbose"
    else:
        verbosity = str(config.get("verbosity", "verbose"))

    if args.min_words is not None:
        min_words = args.min_words
    else:
        cfg_min = config.get("min_words")
        if cfg_min is not None:
            min_words = cfg_min
        else:
            min_words = 1500 if verbosity == "max" else 800 if verbosity == "verbose" else 400

    max_words = args.max_words if args.max_words is not None else config.get("max_words")

    goal_terms: List[str] = []
    if args.cited or config.get("cited", False):
        goal_terms.append("cited")
    if args.grounded or config.get("grounded", False):
        goal_terms.append("grounded")
    if (
        args.verbose
        or args.max_verbose
        or args.min_words is not None
        or verbosity in {"verbose", "max"}
        or config.get("min_words")
    ):
        goal_terms.append("verbose")
    if fresh_days:
        goal_terms.append("fresh")
    if not goal_terms:
        goal_terms = ["cited", "grounded", "verbose"]

    goal_str = ",".join(goal_terms)
    ctx = _prepare_context(
        args.question,
        goal_str,
        router,
        no_bandit=no_bandit,
        log_file=log_file,
    )
    cluster = ctx["router"]["cluster_bias"]

    ctx["data"].update(
        {
            "question": args.question,
            "index_backend": index_backend,
            "k_passages": k_passages,
            "max_chars": max_chars,
            "tags": tags,
            "verbosity": verbosity,
            "min_words": min_words,
        }
    )
    if max_words:
        ctx["data"]["max_words"] = max_words
    if fresh_days:
        ctx["data"]["fresh_days"] = fresh_days

    goal_flags = normalise_goal_flags(goal_str)
    plan_result = plan(
        goal_flags,
        ctx,
        registry,
        interpreter,
        cluster_bias=cluster,
        max_expansions=max_expansions,
    )

    steps = plan_result.get("steps", [])
    if not steps:
        remaining = plan_result.get("remaining_flags") or goal_flags
        print("No plan found. Remaining goals: " + ", ".join(remaining))
        return

    print("Plan:", " -> ".join(behavior for behavior, _ in steps))
    if explain:
        for index, (behavior, info) in enumerate(steps, start=1):
            rationale = info.get("rationale", {})
            why = rationale.get("why", "")
            evidence = info.get("evidence", [])
            effects = info.get("effects", [])
            rewards = ensure_reward_dict(info.get("rewards", {}))
            print(f"{index}. {behavior}")
            if why:
                print(f"   why: {why}")
            if evidence:
                print(f"   evidence: {', '.join(str(ev) for ev in evidence)}")
            if effects:
                print(f"   effects: {', '.join(effects)}")
            reward_preview = {k: round(v, 3) for k, v in rewards.items() if k != "overall"}
            print(f"   rewards: overall={round(rewards.get('overall', 0.0), 3)} {reward_preview}")

    final_ctx = plan_result.get("ctx", ctx)
    final_data = final_ctx.get("data", {})
    final_rewards = ensure_reward_dict(steps[-1][1].get("rewards", {}))
    router.register_outcome(cluster, final_rewards)
    router.register_bandit_outcome(final_ctx, final_rewards)

    answer = final_data.get("answer", "")
    if answer:
        print("Answer\n" + answer)
    sources = final_data.get("sources", [])
    if sources:
        print("Sources:")
        for entry in sources:
            print(f"  {entry['marker']} {entry['source']}")
    print_reward({"rewards": final_rewards})
    if not plan_result.get("goal_satisfied", False):
        remaining = plan_result.get("remaining_flags", [])
        if remaining:
            print("Unmet goal flags:", ", ".join(remaining))

    if log_file:
        router_state = final_ctx.get("router", {})
        record = {
            "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "question": args.question,
            "goal": goal_str,
            "features": router_state.get("bandit_features", []),
            "chosen_cluster": cluster,
            "final_rewards": final_rewards,
            "steps": [behavior for behavior, _ in steps],
            "word_count": final_data.get("word_count"),
            "no_bandit": bool(no_bandit),
        }
        append_jsonl(log_file, record)


def command_plan(args: argparse.Namespace) -> None:
    registry, interpreter, router = build_runtime()
    config = _config_section(args, "plan")

    log_file = args.log_file or config.get("log_file")
    no_bandit = bool(args.no_bandit or config.get("no_bandit", False))
    max_expansions = args.max_expansions if args.max_expansions is not None else config.get("max_expansions", 20)
    explain = bool(args.explain or config.get("explain", False))

    goal_flags = normalise_goal_flags(args.goal)
    if not goal_flags:
        print("No goal flags provided.")
        return

    text_seed = args.text or ""
    ctx = _prepare_context(
        text_seed,
        args.goal,
        router,
        no_bandit=no_bandit,
        log_file=log_file,
    )
    cluster = ctx["router"]["cluster_bias"]
    policies_arg = getattr(args, "policies", "") or ""
    policies = [item.strip() for item in policies_arg.split(",") if item.strip()]
    if policies:
        ctx["data"]["policies"] = policies

    plan_result = plan(
        goal_flags,
        ctx,
        registry,
        interpreter,
        cluster_bias=cluster,
        max_expansions=max_expansions,
    )

    steps = plan_result.get("steps", [])
    if not steps:
        remaining = plan_result.get("remaining_flags") or goal_flags
        print("No plan found. Remaining goals: " + ", ".join(remaining))
        return

    print("Plan:", " -> ".join(behavior for behavior, _ in steps))

    if explain:
        for index, (behavior, info) in enumerate(steps, start=1):
            rationale = info.get("rationale", {})
            why = rationale.get("why", "")
            evidence = info.get("evidence", [])
            effects = info.get("effects", [])
            rewards = ensure_reward_dict(info.get("rewards", {}))
            print(f"{index}. {behavior}")
            if why:
                print(f"   why: {why}")
            if evidence:
                print(f"   evidence: {', '.join(str(ev) for ev in evidence)}")
            if effects:
                print(f"   effects: {', '.join(effects)}")
            reward_preview = {k: round(v, 3) for k, v in rewards.items() if k != "overall"}
            print(f"   rewards: overall={round(rewards.get('overall', 0.0), 3)} {reward_preview}")

    final_ctx = plan_result.get("ctx", ctx)
    final_rewards = ensure_reward_dict(steps[-1][1].get("rewards", {}))
    router.register_outcome(cluster, final_rewards)
    router.register_bandit_outcome(final_ctx, final_rewards)
    print("Final reward:")
    print(json.dumps(final_rewards, indent=2))
    if not plan_result.get("goal_satisfied", False):
        remaining = plan_result.get("remaining_flags", [])
        if remaining:
            print("Unmet goal flags:", ", ".join(remaining))
    if log_file:
        router_state = final_ctx.get("router", {})
        record = {
            "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "goal": args.goal,
            "features": router_state.get("bandit_features", []),
            "chosen_cluster": cluster,
            "final_rewards": final_rewards,
            "steps": [behavior for behavior, _ in steps],
            "unmet_goal_flags": plan_result.get("remaining_flags", []),
            "no_bandit": bool(no_bandit),
        }
        append_jsonl(log_file, record)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Modular behavior CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_HELP_TEXT,
    )
    parser.add_argument("--config", help="Path to configuration file")
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

    ingest_parser = subparsers.add_parser("ingest", help="Ingest web content into the index")
    ingest_group = ingest_parser.add_mutually_exclusive_group(required=True)
    ingest_group.add_argument("--url", help="Fetch a single URL")
    ingest_group.add_argument("--rss", help="Fetch entries from an RSS/Atom feed")
    ingest_group.add_argument("--path", help="Ingest a local text file")
    ingest_parser.add_argument("--tags", help="Comma separated tags to attach", default=None)
    ingest_parser.add_argument("--index-backend", choices=["tfidf", "hnsw"], default=None)
    ingest_parser.add_argument("--limit", type=int, default=None, help="Maximum RSS items to ingest")
    ingest_parser.add_argument("--lang-any", action="store_true", help="Allow non-English content")
    ingest_parser.set_defaults(func=command_ingest)

    ask_parser = subparsers.add_parser("ask", help="Ask questions against ingested knowledge")
    ask_parser.add_argument("--question", required=True)
    ask_parser.add_argument("--k-passages", type=int)
    ask_parser.add_argument("--max-chars", type=int)
    ask_parser.add_argument("--tags", help="Filter retrieval by tags")
    ask_parser.add_argument("--index-backend", choices=["tfidf", "hnsw"], default=None)
    ask_parser.add_argument("--cited", action="store_true", help="Require cited output")
    ask_parser.add_argument("--grounded", action="store_true", help="Require grounded output")
    ask_parser.add_argument("--verbose", action="store_true", help="Prefer verbose output")
    ask_parser.add_argument("--max-verbose", action="store_true", help="Request maximum verbosity")
    ask_parser.add_argument("--min-words", type=int)
    ask_parser.add_argument("--max-words", type=int)
    ask_parser.add_argument("--fresh", type=int, help="Bias toward passages newer than DAYS")
    ask_parser.add_argument("--log-file", help="JSON Lines log output path")
    ask_parser.add_argument("--no-bandit", action="store_true", help="Disable LinUCB routing bias")
    ask_parser.add_argument("--max-expansions", type=int)
    ask_parser.add_argument("--explain", action="store_true")
    ask_parser.set_defaults(func=command_ask)

    plan_parser = subparsers.add_parser("plan", help="Generate a behavior plan")
    plan_parser.add_argument("--goal", required=True)
    plan_parser.add_argument("--text", help="Optional text seed for execution")
    plan_parser.add_argument("--policies", help="Comma separated compliance phrases")
    plan_parser.add_argument("--log-file", help="JSON Lines log output path")
    plan_parser.add_argument("--no-bandit", action="store_true", help="Disable LinUCB routing bias")
    plan_parser.add_argument("--max-expansions", type=int)
    plan_parser.add_argument("--explain", action="store_true")
    plan_parser.set_defaults(func=command_plan)

    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_data = load_config(getattr(args, "config", None))
    setattr(args, "config_data", config_data)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
