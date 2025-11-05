import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from textwrap import dedent
from typing import Any, Dict, List, Mapping, Optional, cast
from uuid import uuid4

from core.audit import log_run
from core.config import load_config
from core import llama_profiles
from core import models as model_store
from core.interfaces import Context
from core.interpreter import Interpreter
from core.planner import normalise_goal_flags, plan
from core.registry import BehaviorRegistry
from core.rewards import ensure_reward_dict
from core.router import SimpleRouter
from core.logs import append_jsonl
from tools import llama_runner
from tools.registry import TOOLS as TOOL_REGISTRY

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

      models [--list] [--show] [--select N] [--select-name NAME]
          Inspect or choose local LLM models discovered via SUPER_MODELS_ROOT. Examples:
            python main.py models --list
            python main.py models --select 2
            python main.py models --select-name llama-3-8b.gguf

      llama --prompt "..." [--model NAME_OR_PATH] [--n-predict N] [--temperature VAL] [--extra ARG ...]
          Run inference through llama.cpp using the active or specified model. Examples:
            python main.py llama --prompt "Hello" --n-predict 64
            python main.py llama --prompt "Summarize this" --model alpha.gguf --temperature 0.7
    """
).strip()


def build_registry() -> BehaviorRegistry:
    return BehaviorRegistry().discover().load_meta()


def build_runtime() -> tuple[BehaviorRegistry, Interpreter, SimpleRouter]:
    registry = build_registry()
    interpreter = Interpreter(registry)
    router = SimpleRouter(registry)
    return registry, interpreter, router


def print_reward(result: Mapping[str, Any]) -> None:
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
    ctx: Context = {"text": text}
    data = _ensure_data(ctx)
    data["text"] = text
    router_state = _ensure_router(ctx)
    router_state["goal_text"] = goal_text
    router_state["no_bandit"] = no_bandit
    if log_file:
        router_state["log_file"] = log_file
    cluster = router.cluster_hint(goal_text, ctx)
    router_state["cluster_bias"] = cluster
    return ctx


def _ensure_data(ctx: Context) -> Dict[str, Any]:
    data = ctx.get("data")
    if not isinstance(data, dict):
        data = {}
        ctx["data"] = data
    data.setdefault("tools_registry", TOOL_REGISTRY)
    return cast(Dict[str, Any], data)


def _ensure_router(ctx: Context) -> Dict[str, Any]:
    router_state = ctx.get("router")
    if not isinstance(router_state, dict):
        router_state = {}
        ctx["router"] = router_state
    return cast(Dict[str, Any], router_state)


def _optional_str(value: object) -> Optional[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _optional_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        try:
            return int(stripped)
        except ValueError:
            return default
    return default


def _config_section(args: argparse.Namespace, section: str) -> Dict[str, object]:
    config = getattr(args, "config_data", None) or {}
    value = config.get(section, {})
    return value if isinstance(value, dict) else {}


def command_summarize(args: argparse.Namespace) -> None:
    registry, interpreter, router = build_runtime()
    ctx = _prepare_context(args.text, args.text, router)
    router_state = _ensure_router(ctx)
    data = _ensure_data(ctx)
    cluster = router_state["cluster_bias"]
    data.update(
        {
            "max_words": args.max_words,
            "max_sentences": args.max_sentences,
            "summary_strategy": args.strategy,
        }
    )
    result = interpreter.execute("summarize", ctx)
    router.register_outcome(cluster, result.get("rewards", {}))
    summary = data.get("summary", "")
    print("Summary:\n" + summary)
    print_reward(result)


def command_format(args: argparse.Namespace) -> None:
    registry, interpreter, router = build_runtime()
    ctx = _prepare_context(args.text, args.text, router)
    router_state = _ensure_router(ctx)
    data = _ensure_data(ctx)
    cluster = router_state["cluster_bias"]
    data.update({"format_style": args.style, "wrap_width": args.wrap_width})
    result = interpreter.execute("document_formatting", ctx)
    router.register_outcome(cluster, result.get("rewards", {}))
    formatted = data.get("formatted_text", "")
    print("Formatted Document:\n" + formatted)
    print_reward(result)


def command_optimize(args: argparse.Namespace) -> None:
    registry, interpreter, router = build_runtime()
    goal_text = f"optimize {args.platform} {args.topic}"
    ctx = _prepare_context(args.text, goal_text, router)
    router_state = _ensure_router(ctx)
    data = _ensure_data(ctx)
    cluster = router_state["cluster_bias"]
    data.update({"platform": args.platform, "topic": args.topic, "max_length": args.max_length})
    result = interpreter.execute("social_post_optimize", ctx)
    router.register_outcome(cluster, result.get("rewards", {}))
    print("Optimized Text:\n" + data.get("optimized_text", ""))
    print("Hashtags:", ", ".join(data.get("hashtags", [])))
    print_reward(result)


def command_check(args: argparse.Namespace) -> None:
    registry, interpreter, router = build_runtime()
    policies = [item.strip() for item in args.policies.split(",") if item.strip()]
    goal_text = "compliance check"
    ctx = _prepare_context(args.text, goal_text, router)
    router_state = _ensure_router(ctx)
    data = _ensure_data(ctx)
    cluster = router_state["cluster_bias"]
    data.update({"policies": policies, "policy_replacement": args.replacement})
    result = interpreter.execute("policy_check", ctx)
    router.register_outcome(cluster, result.get("rewards", {}))
    print("Sanitized Text:\n" + data.get("sanitized_text", ""))
    print("Violations:")
    for violation in data.get("violations", []):
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

    ctx: Context = {}
    payload = _ensure_data(ctx)
    payload.update(data)
    result = interpreter.execute("ingest_web", ctx)
    ingested = payload.get("ingest_log", [])
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
    log_file = args.log_file if isinstance(args.log_file, str) else _optional_str(config.get("log_file"))
    raw_max_exp = args.max_expansions if args.max_expansions is not None else config.get("max_expansions")
    max_expansions = _optional_int(raw_max_exp, 25)
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
    router_state = _ensure_router(ctx)
    data = _ensure_data(ctx)
    cluster = router_state["cluster_bias"]

    data.update(
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
        data["max_words"] = max_words
    if fresh_days:
        data["fresh_days"] = fresh_days

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
    final_data_obj = final_ctx.get("data", {})
    final_data = final_data_obj if isinstance(final_data_obj, dict) else {}
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
        final_router = final_ctx.get("router")
        router_state = final_router if isinstance(final_router, dict) else {}
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
            "meaning": final_data.get("meaning"),
        }
        append_jsonl(Path(log_file), record)


def command_plan(args: argparse.Namespace) -> None:
    registry, interpreter, router = build_runtime()
    config = _config_section(args, "plan")

    log_file = args.log_file if isinstance(args.log_file, str) else _optional_str(config.get("log_file"))
    no_bandit = bool(args.no_bandit or config.get("no_bandit", False))
    raw_max_exp_plan = args.max_expansions if args.max_expansions is not None else config.get("max_expansions")
    max_expansions = _optional_int(raw_max_exp_plan, 20)
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
    router_state = _ensure_router(ctx)
    data = _ensure_data(ctx)
    cluster = router_state["cluster_bias"]
    policies_arg = getattr(args, "policies", "") or ""
    policies = [item.strip() for item in policies_arg.split(",") if item.strip()]
    if policies:
        data["policies"] = policies

    llama_payload: Dict[str, Any] = {}
    config_llama = config.get("llama")
    if isinstance(config_llama, dict):
        llama_payload.update({key: value for key, value in config_llama.items()})
    existing_llama = data.get("llama")
    if isinstance(existing_llama, dict):
        llama_payload.update(existing_llama)

    llama_profile = _optional_str(getattr(args, "llama_profile", None))
    if not llama_profile:
        llama_profile = _optional_str(config.get("llama_profile"))

    llama_vars: Dict[str, str] = {}
    config_llama_vars = config.get("llama_vars")
    if isinstance(config_llama_vars, dict):
        llama_vars.update({str(key): str(value) for key, value in config_llama_vars.items()})
    elif isinstance(config_llama_vars, list):
        llama_vars.update(_parse_profile_vars([str(item) for item in config_llama_vars]))
    elif isinstance(config_llama_vars, str):
        llama_vars.update(_parse_profile_vars([config_llama_vars]))

    config_llama_var = config.get("llama_var")
    if isinstance(config_llama_var, list):
        llama_vars.update(_parse_profile_vars([str(item) for item in config_llama_var]))
    elif isinstance(config_llama_var, str):
        llama_vars.update(_parse_profile_vars([config_llama_var]))

    cli_llama_vars = _parse_profile_vars(getattr(args, "llama_var", None))
    if cli_llama_vars:
        llama_vars.update(cli_llama_vars)

    if llama_profile:
        llama_payload["profile"] = llama_profile

    if llama_vars:
        existing_vars = llama_payload.get("vars")
        merged_vars: Dict[str, str] = {}
        if isinstance(existing_vars, dict):
            merged_vars.update({str(key): str(value) for key, value in existing_vars.items()})
        merged_vars.update(llama_vars)
        llama_payload["vars"] = merged_vars

    if llama_payload:
        data["llama"] = llama_payload
        profile_value = llama_payload.get("profile")
        if isinstance(profile_value, str):
            profile_clean = profile_value.strip()
            if profile_clean:
                data["llama_profile"] = profile_clean
        vars_value = llama_payload.get("vars")
        if isinstance(vars_value, dict):
            data["llama_vars"] = {str(key): str(value) for key, value in vars_value.items()}

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
        final_router = final_ctx.get("router")
        router_state = final_router if isinstance(final_router, dict) else {}
        record = {
            "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "goal": args.goal,
            "features": router_state.get("bandit_features", []),
            "chosen_cluster": cluster,
            "final_rewards": final_rewards,
            "steps": [behavior for behavior, _ in steps],
            "unmet_goal_flags": plan_result.get("remaining_flags", []),
            "no_bandit": bool(no_bandit),
            "meaning": final_ctx.get("data", {}).get("meaning"),
        }
        append_jsonl(Path(log_file), record)


def command_do(args: argparse.Namespace) -> None:
    registry, interpreter, _ = build_runtime()
    task = (args.task or "").strip()
    if not task:
        print("No task provided.")
        return

    workspace = Path(args.workspace or "workspace").resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    perms = _parse_permits(args.permit)
    if not perms:
        perms = {"read"}

    ctx: Context = {"text": task, "dry_run": not args.approve}
    data = _ensure_data(ctx)
    data["task"] = task
    ctx_extra = cast(Dict[str, Any], ctx)
    ctx_extra["perms"] = list(perms)
    ctx_extra["workspace"] = str(workspace)

    parse_result = interpreter.execute("command_parse", ctx)
    if not parse_result.get("ok"):
        print("Could not parse task:", " | ".join(parse_result.get("logs", [])))
        return

    steps = parse_result.get("output", {}).get("steps", [])
    if not steps:
        print("No actionable commands recognized.")
        return

    print("Plan preview:")
    for idx, step in enumerate(steps, start=1):
        behavior = step.get("behavior")
        args_map = step.get("args") or {}
        summary = ", ".join(f"{k}={v}" for k, v in args_map.items())
        print(f"  {idx}. {behavior} {summary}")

    run_id = uuid4().hex
    if not args.approve:
        print("Dry run only. Re-run with --approve to execute.")
        print(f"Run ID: {run_id}")
        meaning_snapshot = ctx.get("data", {}).get("meaning")
        log_run({"mode": "do-dry-run", "task": task, "steps": steps, "run_id": run_id, "meaning": meaning_snapshot})
        return

    execution_logs: List[Dict[str, Any]] = []
    for idx, step in enumerate(steps, start=1):
        behavior = step.get("behavior")
        args_map = step.get("args") or {}
        if not behavior:
            continue
        data_layer = ctx.setdefault("data", {})
        for key, value in args_map.items():
            data_layer[key] = value

        result = interpreter.execute(behavior, ctx)
        logs = result.get("logs", []) or []
        for line in logs:
            print(f"[{behavior}] {line}")
        execution_logs.append(
            {
                "index": idx,
                "behavior": behavior,
                "ok": bool(result.get("ok", True)),
                "logs": logs,
            }
        )
        if not result.get("ok", True):
            print(f"Step {idx} ({behavior}) failed.")
            break

    meaning_snapshot = ctx.get("data", {}).get("meaning")
    log_run({"mode": "do", "task": task, "steps": execution_logs, "run_id": run_id, "meaning": meaning_snapshot})
    print(f"Completed run. Run ID: {run_id}")


def _resolve_llama_model(model_arg: Optional[str]) -> Path:
    return model_store.resolve_model_path(model_arg)


def command_models(args: argparse.Namespace) -> None:
    root = model_store.model_root().resolve()
    selection_performed = False

    try:
        if args.select is not None:
            selected_path = model_store.select_model_by_index(args.select)
            print(f"Selected model #{args.select}: {selected_path}")
            selection_performed = True
        if args.select_name:
            selected_path = model_store.select_model_by_name(args.select_name)
            print(f"Selected model '{args.select_name}': {selected_path}")
            selection_performed = True
    except (FileNotFoundError, IndexError) as exc:
        print(f"Error: {exc}")
        return

    if args.show or selection_performed:
        active = model_store.get_active_model()
        if active:
            print(f"Active model: {active}")
        else:
            print("Active model: (not set)")

    no_flags = not any(
        [
            args.list,
            args.show,
            args.select is not None,
            bool(args.select_name),
        ]
    )
    if args.list or no_flags:
        models = model_store.list_models()
        print(f"Model root: {root}")
        if not models:
            print("No models found. Set SUPER_MODELS_ROOT or add model files.")
            return
        active = model_store.get_active_model()
        active_resolved = active.resolve() if active else None
        for idx, path in enumerate(models, start=1):
            marker = "*" if active_resolved and path.resolve() == active_resolved else " "
            print(f"[{idx:2d}] {marker} {path.name}")


def _parse_profile_vars(items: Optional[List[str]]) -> Dict[str, str]:
    variables: Dict[str, str] = {}
    if not items:
        return variables
    for item in items:
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            continue
        variables[key] = value
    return variables


def command_llama(args: argparse.Namespace) -> None:
    if getattr(args, "list_profiles", False):
        profiles = llama_profiles.list_profiles()
        if not profiles:
            print("No llama profiles configured.")
            return
        print("Available llama profiles:")
        for name, description in profiles:
            suffix = f" - {description}" if description else ""
            print(f"  {name}{suffix}")
        return

    variables = _parse_profile_vars(getattr(args, "var", None))
    profile_settings: Optional[Dict[str, object]] = None
    if getattr(args, "profile", None):
        try:
            profile_settings = llama_profiles.resolve_profile(args.profile, variables)
        except KeyError as exc:
            print(f"Error: {exc}")
            return

    prompt = args.prompt or ""
    if not prompt and args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    if not prompt and profile_settings:
        prompt = str(profile_settings.get("prompt", "") or "")
    if prompt and variables and not profile_settings:
        prompt = llama_profiles.render_template(prompt, variables)
    if not prompt:
        print("Error: provide --prompt/--prompt-file or use a profile with a prompt.")
        return

    model_arg: Optional[str] = args.model
    if not model_arg and profile_settings:
        model_value = profile_settings.get("model")
        if isinstance(model_value, str):
            model_arg = model_value
    try:
        model_path = _resolve_llama_model(model_arg)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return

    n_predict = args.n_predict
    if n_predict is None and profile_settings:
        value = profile_settings.get("n_predict")
        if isinstance(value, int):
            n_predict = value
    temperature = args.temperature
    if temperature is None and profile_settings:
        value = profile_settings.get("temperature")
        if isinstance(value, (int, float)):
            temperature = float(value)

    extra_args: List[str] = []
    if profile_settings:
        extras = profile_settings.get("extra")
        if isinstance(extras, list):
            extra_args.extend(str(item) for item in extras)
    extra_args.extend(args.extra or [])

    try:
        result = llama_runner.run_inference(
            prompt=prompt,
            model=model_path,
            n_predict=n_predict,
            temperature=temperature,
            extra_args=extra_args,
        )
    except llama_runner.LlamaBinaryNotFound as exc:
        print(f"Error: {exc}")
        return

    if result["returncode"] != "0":
        print(f"llama exited with status {result['returncode']}")
        if result["stderr"]:
            print(result["stderr"])
        return

    if args.profile:
        print(f"Profile: {args.profile}")
    print(f"Command: {result['command']}")
    if result["stderr"]:
        print("stderr:\n" + result["stderr"])
    print("Output:\n" + result["stdout"])


def _parse_permits(value: Optional[str]) -> set[str]:
    perms: set[str] = set()
    if not value:
        return perms
    for token in value.split(","):
        token = token.strip().lower()
        if token:
            perms.add(token)
    return perms


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
    plan_parser.add_argument("--llama-profile", help="Apply this llama profile when planning llama_generate steps")
    plan_parser.add_argument("--llama-var", action="append", metavar="KEY=VALUE", help="Template variable for llama profile (repeatable)")
    plan_parser.set_defaults(func=command_plan)

    models_parser = subparsers.add_parser("models", help="Manage local LLM models")
    models_parser.add_argument("--list", action="store_true", help="List models discovered under SUPER_MODELS_ROOT")
    models_parser.add_argument("--show", action="store_true", help="Show the currently active model")
    models_parser.add_argument("--select", type=int, help="Select model by 1-based index from the list")
    models_parser.add_argument("--select-name", help="Select model by exact or partial name")
    models_parser.set_defaults(func=command_models)

    llama_parser = subparsers.add_parser("llama", help="Run inference using llama.cpp")
    llama_prompt_group = llama_parser.add_mutually_exclusive_group(required=False)
    llama_prompt_group.add_argument("--prompt", help="Prompt text to send to the model")
    llama_prompt_group.add_argument("--prompt-file", help="Path to a file containing the prompt")
    llama_parser.add_argument("--model", help="Model name (from models list) or direct path")
    llama_parser.add_argument("--profile", help="Use a named llama profile")
    llama_parser.add_argument("--list-profiles", action="store_true", help="List available llama profiles and exit")
    llama_parser.add_argument("--var", action="append", metavar="KEY=VALUE", help="Template variable for the profile (repeatable)")
    llama_parser.add_argument("--n-predict", type=int, help="Number of tokens to generate")
    llama_parser.add_argument("--temperature", type=float, help="Sampling temperature")
    llama_parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help="Additional argument forwarded to llama.cpp (repeat for each flag, e.g. --extra --simple-io)",
    )
    llama_parser.set_defaults(func=command_llama)

    do_parser = subparsers.add_parser("do", help="Run automation commands via behaviors")
    do_parser.add_argument("--task", required=True, help="Natural-language automation instruction")
    do_parser.add_argument("--workspace", default="workspace", help="Workspace directory for file actions")
    do_parser.add_argument(
        "--permit",
        default="",
        help="Comma separated permissions to grant (read,write,net,exec)",
    )
    do_parser.add_argument("--approve", action="store_true", help="Execute instead of dry-run preview")
    do_parser.set_defaults(func=command_do)

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
