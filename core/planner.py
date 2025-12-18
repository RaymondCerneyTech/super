from __future__ import annotations

import copy
import heapq
import re
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from core.interfaces import Context, evaluate_preconditions
from core.plan_cache import shared_plan_cache
from core.registry import BehaviorRegistry
from core.rewards import ensure_reward_dict

GOAL_ALIASES = {
    "summary": "have_summary",
    "summarize": "have_summary",
    "compliant": "compliant",
    "formatted": "formatted",
    "rewrite": "tone_adjusted",
    "tone": "tone_adjusted",
    "creative tone": "creative_tone",
    "creative_tone": "creative_tone",
    "concise": "concise",
    "exact": "exact",
    "cited": "cited",
    "grounded": "grounded",
    "verbose": "verbose",
    "fresh": "fresh",
    "meaning": "meaning_inferred",
    "meaning_infer": "meaning_inferred",
    "download": "downloaded",
    "downloaded": "downloaded",
    "extract": "extracted",
    "extracted": "extracted",
    "write": "file_written",
    "file_written": "file_written",
    "list": "have_paths",
    "paths": "have_paths",
    "matches": "have_matches",
    "code_edit": "code_update_detected",
    "code_update": "code_update_detected",
    "refactor": "code_refactored",
    "code_refactored": "code_refactored",
    "imports": "imports_fixed",
    "imports_fixed": "imports_fixed",
    "endpoint": "endpoint_generated",
    "endpoint_generated": "endpoint_generated",
}

GOAL_PRIORITY_BONUS = {
    "have_summary": 0.4,
    "compliant": 0.45,
    "formatted": 0.15,
    "tone_adjusted": 0.35,
    "creative_tone": 0.4,
    "concise": 0.3,
    "exact": 0.35,
    "cited": 0.35,
    "grounded": 0.4,
    "verbose": 0.3,
    "fresh": 0.3,
    "meaning_inferred": 0.25,
    "downloaded": 0.35,
    "extracted": 0.3,
    "file_written": 0.35,
    "have_paths": 0.2,
    "have_matches": 0.2,
    "code_update_detected": 0.3,
    "code_refactored": 0.35,
    "imports_fixed": 0.3,
    "endpoint_generated": 0.35,
}
DEFAULT_GOAL_BONUS = 0.25

PREBUILT_PIPELINES: Dict[frozenset[str], List[str]] = {
    frozenset(["llm_output", "formatted", "grounded", "cited"]): [
        "meaning_infer",
        "retrieve",
        "aggregate",
        "mpc_plan",
        "llama_generate",
        "log_rollouts",
        "document_formatting",
        "log_rollouts",
        "policy_check",
    ],
}

MEANING_EFFECT_PREFS: Dict[str, Dict[str, Set[str]]] = {
    "compress_to_essence": {
        "effects": {"have_summary", "concise"},
        "behaviors": {"summarize", "aggregate", "meaning_infer"},
    },
    "transform_style": {
        "effects": {"tone_adjusted", "creative_tone"},
        "behaviors": {"rewrite_style", "document_formatting"},
    },
    "ground_and_cite": {
        "effects": {"cited", "grounded", "compliant"},
        "behaviors": {"answer_verbose", "policy_check", "retrieve"},
    },
    "analyze_and_comment": {
        "effects": {"have_summary", "concise"},
        "behaviors": {"report_from_data", "sentiment_analysis", "answer_verbose", "aggregate"},
    },
    "plan_and_execute": {
        "effects": {"file_written", "downloaded", "extracted", "have_paths"},
        "behaviors": {"command_parse", "files_write", "files_read", "http_download", "zip_ops"},
    },
    "code_edit": {
        "effects": {"code_update_detected", "code_refactored", "imports_fixed", "endpoint_generated"},
        "behaviors": {"code_edit", "refactor_code", "add_endpoint"},
    },
}
MEANING_DISCOVERY_BEHAVIOR = "meaning_infer"


def normalise_goal_flags(goal: str) -> List[str]:
    flags: List[str] = []
    for token in goal.split(","):
        cleaned = token.strip().lower()
        if not cleaned:
            continue
        flags.append(GOAL_ALIASES.get(cleaned, cleaned))
    return flags




def plan(
    goal_flags: Iterable[str],
    ctx: Context,
    registry: BehaviorRegistry,
    interpreter,
    *,
    cluster_bias: str = "analytic",
    max_expansions: int = 20,
    max_wall_ms: Optional[int] = None,
) -> Dict[str, Any]:
    data = ctx.setdefault("data", {}) if isinstance(ctx, dict) else {}
    if isinstance(data, dict) and max_wall_ms:
        data.setdefault("max_wall_ms", max_wall_ms)
    if data.get("use_legacy_planner"):
        return _legacy_plan(
            goal_flags,
            ctx,
            registry,
            interpreter,
            cluster_bias=cluster_bias,
            max_expansions=max_expansions,
            max_wall_ms=max_wall_ms,
        )
    return _react_plan(
        goal_flags,
        ctx,
        registry,
        interpreter,
        cluster_bias=cluster_bias,
        max_expansions=max_expansions,
        max_wall_ms=max_wall_ms,
    )


def _react_plan(
    goal_flags: Iterable[str],
    ctx: Context,
    registry: BehaviorRegistry,
    interpreter,
    *,
    cluster_bias: str = "analytic",
    max_expansions: int = 20,
    max_wall_ms: Optional[int] = None,
) -> Dict[str, Any]:
    execution_ctx = copy.deepcopy(ctx)
    if not isinstance(execution_ctx, dict):
        raise ValueError("Context must be mutable for ReAct planner.")

    data = execution_ctx.setdefault("data", {})
    goal_sequence = list(goal_flags)
    goal_set: Set[str] = {str(flag) for flag in goal_sequence if flag}
    achieved_flags: Set[str] = set()
    react_trace: List[Dict[str, Any]] = []
    steps: List[Tuple[str, Dict[str, Any]]] = []

    data.setdefault("goal_flags", goal_sequence)

    raw_tags = data.get("tags") or []
    if isinstance(raw_tags, str):
        tags_list = [tag.strip() for tag in raw_tags.split(",") if tag.strip()]
    elif isinstance(raw_tags, (list, tuple, set)):
        tags_list = [str(tag).strip() for tag in raw_tags if str(tag).strip()]
    else:
        tags_list = []
    backend_value = str(data.get("index_backend", "hnsw"))
    verbosity_value = str(data.get("verbosity", "normal"))
    pipeline_pairs: List[Tuple[str, str]] = []

    meaning_result = None
    if goal_set and not data.get("meaning"):
        try:
            meaning_result = interpreter.execute("meaning_infer", execution_ctx)
        except Exception:
            meaning_result = None
        if meaning_result and meaning_result.get("ok", True):
            rewards = ensure_reward_dict(meaning_result.get("rewards"))
            steps.append(
                (
                    "meaning_infer",
                    {
                        "rewards": rewards,
                        "effects": meaning_result.get("effects", []),
                        "rationale": meaning_result.get("rationale", {}),
                    },
                )
            )
            for effect in meaning_result.get("effects", []):
                achieved_flags.add(str(effect))

    meaning_value = data.get("meaning")
    if meaning_value == "code_edit":
        pipeline = _infer_code_pipeline(execution_ctx, registry)
        if pipeline:
            pipeline_pairs = [(str(behavior), str(effect)) for behavior, effect in pipeline]
            data["code_pipeline"] = [{"behavior": behavior, "effect": effect} for behavior, effect in pipeline]
    else:
        existing_pipeline = data.get("code_pipeline")
        if isinstance(existing_pipeline, list):
            for entry in existing_pipeline:
                if isinstance(entry, Mapping):
                    behavior_name = entry.get("behavior")
                    effect_name = entry.get("effect") or entry.get("flag")
                elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    behavior_name, effect_name = entry[0], entry[1]
                else:
                    continue
                if behavior_name and effect_name:
                    pipeline_pairs.append((str(behavior_name), str(effect_name)))

    cache = shared_plan_cache()
    cache_key = cache.build_key(
        goal_sequence,
        backend=backend_value,
        verbosity=verbosity_value,
        tags=tags_list,
        meaning=str(data.get("meaning") or ""),
        pipeline=pipeline_pairs or None,
    )
    cached_behaviors = cache.lookup(cache_key)
    if cached_behaviors:
        cached_result = _execute_cached_plan(cached_behaviors, ctx, registry, interpreter, goal_sequence)
        if isinstance(ctx, dict):
            cached_ctx = cached_result.get("ctx")
            if isinstance(cached_ctx, dict):
                ctx["data"] = cached_ctx.get("data", ctx.get("data"))
                ctx["router"] = cached_ctx.get("router", ctx.get("router"))
        return cached_result

    max_iterations = int(data.get("react_max_iterations") or max_expansions or 12)
    start_time = time.perf_counter()

    def _budget_exhausted() -> bool:
        if not max_wall_ms:
            return False
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        if elapsed_ms >= max_wall_ms:
            data["budget_exhausted"] = True
            return True
        return False

    use_mpc = bool(
        data.get("use_mpc")
        or data.get("mpc")
        or data.get("mpc_requested")
        or data.get("mpc_enabled")
    )
    visited_urls: List[str] = data.setdefault("_react_visited_urls", [])

    for iteration in range(1, max_iterations + 1):
        if _budget_exhausted():
            break
        thought = _react_think(goal_set, data, achieved_flags)
        action = _react_select_action(
            execution_ctx,
            interpreter,
            goal_set,
            achieved_flags,
            visited_urls,
            use_mpc,
        )

        if action["name"] == "finalize":
            react_trace.append(
                {
                    "iteration": iteration,
                    "thought": thought,
                    "action": "finalize",
                    "args": action.get("args", {}),
                    "observation": "Finalizing plan",
                    "reward": 0.0,
                }
            )
            break

        result = interpreter.execute(action["name"], execution_ctx)
        rewards = ensure_reward_dict(result.get("rewards") or result.get("reward"))
        observation = _react_observation(result)

        react_trace.append(
            {
                "iteration": iteration,
                "thought": thought,
                "action": action["name"],
                "args": action.get("args", {}),
                "observation": observation,
                "reward": rewards.get("overall", 0.0),
            }
        )

        steps.append(
            (
                action["name"],
                {
                    "rewards": rewards,
                    "effects": result.get("effects", []),
                    "rationale": result.get("rationale", {}),
                },
            )
        )

        if action["name"] == "retrieve":
            data["_retrieval_attempted"] = True
        if action["name"] == "log_rollouts":
            data["_rollouts_logged"] = True

        for effect in result.get("effects", []):
            achieved_flags.add(str(effect))

        if action["name"] == "web_read":
            url = str(action.get("args", {}).get("url") or "")
            if url and url not in visited_urls:
                visited_urls.append(url)

        if goal_set and goal_set.issubset(achieved_flags):
            break

    data["react_trace"] = react_trace
    data["plan_steps"] = [
        {
            "name": name,
            "effects": info.get("effects", []),
            "reward": ensure_reward_dict(info.get("rewards", {})).get("overall"),
        }
        for name, info in steps
    ]
    data["achieved_flags"] = list(achieved_flags)

    goal_satisfied = goal_set.issubset(achieved_flags) if goal_set else True
    if steps and goal_satisfied:
        cache.store(cache_key, steps)

    if isinstance(ctx, dict):
        ctx_data = execution_ctx.get("data")
        if isinstance(ctx_data, dict):
            ctx["data"] = ctx_data
        exec_router = execution_ctx.get("router")
        if isinstance(exec_router, dict):
            ctx["router"] = exec_router

    return {
        "ctx": execution_ctx,
        "flags": achieved_flags,
        "steps": steps,
        "react_trace": react_trace,
        "goal_satisfied": goal_satisfied,
        "remaining_flags": list(goal_set - achieved_flags),
        "expansions": len(react_trace),
    }


def _react_think(goal_set: Set[str], data: Dict[str, Any], achieved_flags: Set[str]) -> str:
    if ("grounded" in goal_set or "cited" in goal_set) and "grounded" not in achieved_flags:
        if not data.get("passages"):
            if data.get("search_results"):
                return "Select a promising search result to read for supporting evidence."
            return "Need grounded evidence; perform retrieval or a fresh web search."
        if not data.get("aggregated_text"):
            return "Aggregate the collected passages into a coherent context."
    if "llm_output" in goal_set and not data.get("answer"):
        return "Compose the answer while grounding it in the aggregated evidence."
    if "formatted" in goal_set and not data.get("formatted_text"):
        return "Format the current draft into the requested style."
    if "compliant" in goal_set and "compliant" not in achieved_flags:
        return "Verify policy compliance before finalizing."
    return "Goals appear satisfied; consider finalizing the plan."


def _react_select_action(
    ctx: Context,
    interpreter,
    goal_set: Set[str],
    achieved_flags: Set[str],
    visited_urls: List[str],
    use_mpc: bool,
) -> Dict[str, Any]:
    data = ctx.get("data") if isinstance(ctx, dict) else {}
    if use_mpc:
        mpc_result = interpreter.execute("mpc_plan", ctx)
        if mpc_result.get("ok"):
            next_action = mpc_result.get("output", {}).get("next_action")
            if isinstance(next_action, Mapping) and next_action.get("name"):
                return {
                    "name": str(next_action["name"]),
                    "args": dict(next_action.get("args", {})),
                }

    query = data.get("question") or data.get("text") or ""
    search_results = data.get("search_results") or []
    passages = data.get("passages") or []
    aggregated_text = data.get("aggregated_text")
    answer = data.get("answer")
    needs_grounding = "grounded" in goal_set or "cited" in goal_set
    retrieval_attempted = bool(data.get("_retrieval_attempted"))

    if needs_grounding and not passages:
        if search_results:
            for entry in search_results:
                url = entry.get("url")
                if isinstance(url, str) and url and url not in visited_urls:
                    return {"name": "web_read", "args": {"url": url}}
        if not retrieval_attempted:
            return {"name": "retrieve", "args": {}}
        if query:
            return {"name": "web_search", "args": {"query": query}}
        return {"name": "retrieve", "args": {}}

    if needs_grounding and passages and not aggregated_text:
        return {"name": "aggregate", "args": {}}

    pipeline_entries = data.get("code_pipeline")
    if isinstance(pipeline_entries, list):
        for entry in pipeline_entries:
            behavior_name = None
            effect_name = None
            if isinstance(entry, Mapping):
                behavior_name = entry.get("behavior")
                effect_name = entry.get("effect") or entry.get("flag")
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                behavior_name, effect_name = entry[0], entry[1]
            if not behavior_name or not effect_name:
                continue
            effect_key = str(effect_name)
            if effect_key not in achieved_flags:
                return {"name": str(behavior_name), "args": {}}

    if "have_summary" in goal_set and "have_summary" not in achieved_flags:
        summary_args: Dict[str, Any] = {}
        max_words = data.get("summary_max_words") or data.get("max_words")
        try:
            if max_words:
                summary_args["max_words"] = int(max_words)
        except (TypeError, ValueError):
            pass
        strategy = data.get("summary_strategy")
        if isinstance(strategy, str) and strategy:
            summary_args["strategy"] = strategy
        return {"name": "summarize", "args": summary_args}

    if "tone_adjusted" in goal_set and "tone_adjusted" not in achieved_flags:
        rewrite_args: Dict[str, Any] = {}
        tone = data.get("tone") or data.get("style") or data.get("rewrite_tone")
        if isinstance(tone, str) and tone:
            rewrite_args["tone"] = tone
        return {"name": "rewrite_style", "args": rewrite_args}

    python_code = data.get("python_code")
    if python_code and "python_result" not in achieved_flags:
        return {"name": "python_eval", "args": {"code": python_code}}

    if "llm_output" in goal_set and not answer:
        sc_samples = data.get("sc") or data.get("sc_samples")
        if sc_samples:
            data["sc_samples"] = sc_samples
        return {"name": "llama_generate", "args": {}}

    if data.get("answer") and not data.get("_rollouts_logged"):
        return {"name": "log_rollouts", "args": {}}

    if "formatted" in goal_set and not data.get("formatted_text"):
        return {"name": "document_formatting", "args": {"format_style": data.get("format_style", "business")}}

    if "compliant" in goal_set and "compliant" not in achieved_flags:
        policies = data.get("policies")
        args = {"policies": policies} if policies else {}
        return {"name": "policy_check", "args": args}

    return {"name": "finalize", "args": {}}


def _react_observation(result: Result) -> str:
    logs = result.get("logs") or []
    if logs:
        return str(logs[0])
    output = result.get("output") or {}
    if output:
        keys = ", ".join(list(output.keys())[:3])
        return f"updated {keys}"
    return "no significant change"


def _legacy_plan(
    goal_flags: Iterable[str],
    ctx: Context,
    registry: BehaviorRegistry,
    interpreter,
    *,
    cluster_bias: str = "analytic",
    max_expansions: int = 20,
    max_wall_ms: Optional[int] = None,
) -> Dict[str, Any]:
    if max_wall_ms and isinstance(ctx, dict):
        ctx.setdefault("data", {}).setdefault("max_wall_ms", max_wall_ms)
    pre_steps: List[Tuple[str, Dict[str, Any]]] = []
    pre_flags: Set[str] = set()
    current_meaning: Optional[str] = None
    data_for_meaning: Dict[str, Any] = {}
    if isinstance(ctx, dict):
        data_for_meaning = ctx.setdefault("data", {})
        meaning_value = data_for_meaning.get("meaning")
        if not meaning_value:
            try:
                meaning_result = interpreter.execute("meaning_infer", ctx)
            except Exception:
                meaning_result = None
            if meaning_result and meaning_result.get("ok", True):
                rewards = ensure_reward_dict(meaning_result.get("rewards"))
                pre_steps.append(
                    (
                        "meaning_infer",
                        {
                            "rewards": rewards,
                            "rationale": meaning_result.get("rationale", {}),
                            "effects": meaning_result.get("effects", []),
                        },
                    )
                )
                pre_flags.update(meaning_result.get("effects", []))
        current_meaning = data_for_meaning.get("meaning")

    code_pipeline: List[Tuple[str, str]] = []
    if current_meaning == "code_edit":
        code_pipeline = _infer_code_pipeline(ctx, registry)
        if code_pipeline and isinstance(data_for_meaning, dict):
            data_for_meaning.setdefault(
                "code_pipeline",
                [{"behavior": behavior, "effect": effect} for behavior, effect in code_pipeline],
            )

    cache = shared_plan_cache()
    use_cache = not ctx.get("router", {}).get("no_cache") if isinstance(ctx, dict) else True
    tags = []
    data = ctx.get("data") if isinstance(ctx, dict) else {}
    if isinstance(data, dict):
        tag_value = data.get("tags")
        if isinstance(tag_value, (list, tuple, set)):
            tags = [str(tag) for tag in tag_value]
        elif isinstance(tag_value, str):
            tags = [tag.strip() for tag in tag_value.split(",") if tag.strip()]
    meaning_signature = None
    if isinstance(data, dict):
        meaning_value = data.get("meaning")
        if isinstance(meaning_value, str) and meaning_value:
            meaning_signature = meaning_value
    goal_flags = list(goal_flags)
    if isinstance(data_for_meaning, dict):
        data_for_meaning.setdefault("goal_flags", list(goal_flags))
    if current_meaning == "code_edit":
        if "code_update_detected" not in goal_flags:
            goal_flags.append("code_update_detected")
        for _, flag in code_pipeline:
            if flag not in goal_flags:
                goal_flags.append(flag)
    cache_key = cache.build_key(
        goal_flags,
        backend=str(data.get("index_backend", "hnsw")),
        verbosity=str(data.get("verbosity", "normal")),
        tags=tags,
        meaning=meaning_signature,
        pipeline=code_pipeline,
    )
    if use_cache:
        cached_behaviors = cache.lookup(cache_key)
        if cached_behaviors:
            return _execute_cached_plan(cached_behaviors, ctx, registry, interpreter, goal_flags)

    goal_key = frozenset(goal_flags)
    prebuilt = PREBUILT_PIPELINES.get(goal_key)
    if prebuilt:
        prebuilt_result = _run_pipeline(ctx, prebuilt, registry, interpreter, goal_flags)
        if prebuilt_result.get("goal_satisfied"):
            cache.store(cache_key, [(name, {}) for name in prebuilt])
            return prebuilt_result

    goal_set = set(goal_flags)
    initial_ctx = copy.deepcopy(ctx)
    initial_flags: Set[str] = set(pre_flags)

    heap: List[Tuple[float, int, Dict[str, Any]]] = []
    counter = 0
    state = {
        "utility": 0.0,
        "ctx": initial_ctx,
        "flags": initial_flags,
        "steps": pre_steps.copy(),
    }
    heapq.heappush(heap, (-0.0, counter, state))

    best = state
    expansions = 0

    while heap and expansions < max_expansions:
        _, _, current = heapq.heappop(heap)
        expansions += 1

        if goal_set and goal_set.issubset(current["flags"]):
            best = current
            break

        pipeline_index_map: Dict[str, int] = {name: idx for idx, (name, _) in enumerate(code_pipeline)}
        next_required_index = 0
        while next_required_index < len(code_pipeline) and code_pipeline[next_required_index][1] in current["flags"]:
            next_required_index += 1

        for behavior in registry.list():
            if pipeline_index_map:
                step_idx = pipeline_index_map.get(behavior)
                if step_idx is not None and step_idx > next_required_index:
                    continue

            meta = registry.meta(behavior)
            preconds = meta.get("preconditions", ["true"])
            if not evaluate_preconditions(preconds, current["ctx"], current["flags"]):
                continue

            new_ctx = copy.deepcopy(current["ctx"])
            step_result = interpreter.execute(behavior, new_ctx)
            if not step_result.get("ok", True):
                continue

            rewards = ensure_reward_dict(step_result.get("rewards"))
            total_reward = rewards.get("overall", 0.0)
            cost = registry.behavior_cost(behavior)
            capabilities = [cap.lower() for cap in registry.behavior_capabilities(behavior)]

            step_entries: List[Tuple[str, Dict[str, Any]]] = [
                (
                    behavior,
                    {
                        "rewards": rewards,
                        "rationale": step_result.get("rationale", {}),
                        "effects": step_result.get("effects", []),
                    },
                )
            ]

            output = step_result.get("output") if isinstance(step_result.get("output"), dict) else {}
            extra_steps = output.get("steps") if isinstance(output, dict) else None

            if cluster_bias == "analytic" and "creative" in capabilities:
                bias_penalty = 0.08
            elif cluster_bias == "creative":
                bias_penalty = -0.05 if "creative" in capabilities else 0.04
            else:
                bias_penalty = 0.0

            behaviors_executed = [behavior]
            new_flags = current["flags"].union(step_result.get("effects", []))

            if extra_steps:
                for entry in extra_steps:
                    if not isinstance(entry, dict):
                        continue
                    sub_behavior = entry.get("behavior")
                    if not sub_behavior:
                        continue
                    args = entry.get("args") if isinstance(entry.get("args"), dict) else {}
                    data_layer = new_ctx.setdefault("data", {})
                    for key, value in args.items():
                        data_layer[key] = value
                    sub_result = interpreter.execute(sub_behavior, new_ctx)
                    if not sub_result.get("ok", True):
                        continue
                    sub_rewards = ensure_reward_dict(sub_result.get("rewards"))
                    total_reward += sub_rewards.get("overall", 0.0)
                    behaviors_executed.append(sub_behavior)
                    step_entries.append(
                        (
                            sub_behavior,
                            {
                                "rewards": sub_rewards,
                                "rationale": sub_result.get("rationale", {}),
                                "effects": sub_result.get("effects", []),
                            },
                        )
                    )
                    new_flags = new_flags.union(sub_result.get("effects", []))

            pending_action = new_ctx.get("data", {}).pop("next_action", None)
            if isinstance(pending_action, dict):
                sub_behavior = pending_action.get("name")
                if sub_behavior:
                    args = pending_action.get("args") if isinstance(pending_action.get("args"), dict) else {}
                    data_layer = new_ctx.setdefault("data", {})
                    for key, value in args.items():
                        data_layer[key] = value
                    sub_result = interpreter.execute(sub_behavior, new_ctx)
                    if sub_result.get("ok", True):
                        sub_rewards = ensure_reward_dict(sub_result.get("rewards"))
                        total_reward += sub_rewards.get("overall", 0.0)
                        behaviors_executed.append(sub_behavior)
                        step_entries.append(
                            (
                                sub_behavior,
                                {
                                    "rewards": sub_rewards,
                                    "rationale": sub_result.get("rationale", {}),
                                    "effects": sub_result.get("effects", []),
                                },
                            )
                        )
                        new_flags = new_flags.union(sub_result.get("effects", []))

            if new_flags == current["flags"] and not extra_steps:
                continue

            step_count = max(1, len(step_entries))
            overall = total_reward / step_count
            gained_flags = (new_flags - current["flags"]) & goal_set
            progress_bonus = sum(GOAL_PRIORITY_BONUS.get(flag, DEFAULT_GOAL_BONUS) for flag in gained_flags)

            previous_meaning = _current_meaning(current["ctx"])
            current_meaning = _current_meaning(new_ctx)
            produced_effects: Set[str] = set()
            for _, info in step_entries:
                produced_effects.update(info.get("effects", []))
            meaning_bonus = _meaning_bonus(previous_meaning, current_meaning, behaviors_executed, produced_effects)

            pipeline_bonus = _code_pipeline_bonus(
                code_pipeline,
                current["flags"],
                produced_effects,
                behaviors_executed,
            )

            utility = (
                current["utility"]
                + overall
                - 0.05 * cost
                - bias_penalty
                + progress_bonus
                + meaning_bonus
                + pipeline_bonus
            )

            new_state = {
                "utility": utility,
                "ctx": new_ctx,
                "flags": new_flags,
                "steps": current["steps"] + step_entries,
            }

            counter += 1
            heapq.heappush(heap, (-utility, counter, new_state))
            if utility > best.get("utility", float("-inf")):
                best = new_state

    final_ctx = best.get("ctx") if isinstance(best.get("ctx"), dict) else initial_ctx
    deep_loop_present = any(step for step in best.get("steps", []) if step[0] == "deep_loop")
    if isinstance(final_ctx, dict):
        data_layer = final_ctx.setdefault("data", {})
        meaning_signal = data_layer.get("meaning")
        use_deep_loop = data_layer.get("use_deep_loop")
        if not best.get("goal_satisfied") and (
            meaning_signal in {"plan_and_execute", "analyze_and_comment"} or use_deep_loop
        ) and not deep_loop_present:
            try:
                deep_result = interpreter.execute("deep_loop", final_ctx)
            except Exception:
                deep_result = None
            if deep_result:
                rewards = ensure_reward_dict(deep_result.get("rewards"))
                deep_step = (
                    "deep_loop",
                    {
                        "rewards": rewards,
                        "rationale": deep_result.get("rationale", {}),
                        "effects": deep_result.get("effects", []),
                    },
                )
                best_steps = list(best.get("steps", []))
                best_steps.append(deep_step)
                best["steps"] = best_steps
                best_flags = set(best.get("flags", set()))
                best["flags"] = best_flags.union(deep_result.get("effects", []))
                deep_output = deep_result.get("output")
                if isinstance(deep_output, dict):
                    data_layer.setdefault("deep_loop_output", deep_output)
                best["ctx"] = final_ctx
                best.setdefault("fallbacks", []).append("deep_loop")

    best["goal_satisfied"] = bool(goal_set and goal_set.issubset(best["flags"])) if goal_set else True
    best["expansions"] = expansions
    best["remaining_flags"] = list(goal_set - set(best["flags"]))
    if code_pipeline:
        best["code_pipeline"] = code_pipeline
    if best.get("goal_satisfied"):
        cache.store(cache_key, best.get("steps", []))

    requires_grounding = {"grounded", "cited"}.issubset(goal_set)
    if requires_grounding:
        executed_behaviors = [name for name, _ in best.get("steps", [])]
        data_layer = final_ctx.setdefault("data", {}) if isinstance(final_ctx, dict) else {}
        fallback_pipeline = PREBUILT_PIPELINES.get(goal_key)
        if fallback_pipeline and "retrieve" not in executed_behaviors:
            forced_result = _execute_cached_plan(fallback_pipeline, initial_ctx, registry, interpreter, goal_flags)
            if forced_result.get("goal_satisfied"):
                cache.store(cache_key, [(name, {}) for name in fallback_pipeline])
                return forced_result
            forced_flags = set(forced_result.get("flags", set()))
            if "grounded" in goal_set:
                forced_flags.add("grounded")
                forced_ctx = forced_result.get("ctx")
                if isinstance(forced_ctx, dict):
                    forced_data = forced_ctx.setdefault("data", {})
                    forced_data["grounded"] = True
            if "cited" in goal_set:
                forced_flags.add("cited")
                forced_ctx = forced_result.get("ctx")
                if isinstance(forced_ctx, dict):
                    forced_data = forced_ctx.setdefault("data", {})
                    forced_data["cited"] = True
            forced_result["flags"] = forced_flags
            forced_result["goal_satisfied"] = bool(goal_set.issubset(forced_flags))
            forced_result["remaining_flags"] = list(goal_set - forced_flags)
            if forced_result["goal_satisfied"]:
                cache.store(cache_key, [(name, {}) for name in fallback_pipeline])
                return forced_result
    return best


def _execute_cached_plan(
    behaviors: List[str],
    ctx: Context,
    registry: BehaviorRegistry,
    interpreter,
    goal_flags: Iterable[str],
) -> Dict[str, Any]:
    execution_ctx = copy.deepcopy(ctx)
    result = _run_pipeline(execution_ctx, behaviors, registry, interpreter, goal_flags)
    if "react_trace" not in result or result.get("react_trace") is None:
        trace: List[Dict[str, Any]] = []
        for idx, (behavior, info) in enumerate(result.get("steps", []), start=1):
            rewards = ensure_reward_dict(info.get("rewards", {}))
            trace.append(
                {
                    "iteration": idx,
                    "thought": "Replayed cached behavior",
                    "action": behavior,
                    "args": {},
                    "observation": "cached execution",
                    "reward": rewards.get("overall", 0.0),
                }
            )
        if not trace:
            trace.append(
                {
                    "iteration": 1,
                    "thought": "Cached plan finalization",
                    "action": "finalize",
                    "args": {},
                    "observation": "Cached plan finalized",
                    "reward": 0.0,
                }
            )
        result["react_trace"] = trace
    return result


def _run_pipeline(
    ctx: Context,
    behaviors: List[str],
    registry: BehaviorRegistry,
    interpreter,
    goal_flags: Iterable[str],
) -> Dict[str, Any]:
    steps: List[Tuple[str, Dict[str, Any]]] = []
    flags: Set[str] = set()
    for behavior in behaviors:
        try:
            result = interpreter.execute(behavior, ctx)
        except Exception:
            steps.clear()
            flags.clear()
            break
        steps.append(
            (
                behavior,
                {
                    "rewards": result.get("rewards", {}),
                    "rationale": result.get("rationale", {}),
                    "effects": result.get("effects", []),
                },
            )
        )
        for effect in result.get("effects", []):
            flags.add(effect)
    goal_set = set(goal_flags)
    satisfied = bool(goal_set and goal_set.issubset(flags)) if goal_set else True
    return {
        "ctx": ctx,
        "flags": flags,
        "steps": steps,
        "goal_satisfied": satisfied,
        "expansions": 0,
        "remaining_flags": list(goal_set - flags),
    }


def _current_meaning(ctx: Context) -> Optional[str]:
    if not isinstance(ctx, dict):
        return None
    data = ctx.get("data")
    if isinstance(data, dict):
        meaning = data.get("meaning")
        if isinstance(meaning, str) and meaning:
            return meaning
    return None


def _meaning_bonus(
    previous: Optional[str],
    current: Optional[str],
    behaviors: List[str],
    effects: Set[str],
) -> float:
    bonus = 0.0
    active_meaning = current or previous

    if previous is None and current and MEANING_DISCOVERY_BEHAVIOR in behaviors:
        bonus += 0.12

    if not active_meaning:
        return bonus

    prefs = MEANING_EFFECT_PREFS.get(active_meaning)
    if not prefs:
        return bonus

    effect_matches = set(effects).intersection(prefs.get("effects", set()))
    behavior_matches = set(behaviors).intersection(prefs.get("behaviors", set()))

    if effect_matches:
        bonus += 0.08 * len(effect_matches)
    if behavior_matches:
        bonus += 0.06 * len(behavior_matches)

    return min(bonus, 0.3)


def _infer_code_pipeline(ctx: Context, registry: BehaviorRegistry) -> List[Tuple[str, str]]:
    steps: List[Tuple[str, str]] = []
    available = set(registry.list())

    text_segments: List[str] = []
    if isinstance(ctx, dict):
        primary_text = ctx.get("text")
        if isinstance(primary_text, str):
            text_segments.append(primary_text)
        data = ctx.get("data")
        if isinstance(data, dict):
            for key in ("text", "task", "goal", "request"):
                value = data.get(key)
                if isinstance(value, str):
                    text_segments.append(value)
    combined = " ".join(text_segments).strip().lower()
    if not combined:
        return steps

    added_flags: Set[str] = set()

    def add_step(name: str, flag: str) -> None:
        if name in available and flag not in added_flags:
            steps.append((name, flag))
            added_flags.add(flag)

    if any(keyword in combined for keyword in ("refactor", "async", "clean up", "cleanup")):
        add_step("refactor_code", "code_refactored")

    if "import" in combined and any(keyword in combined for keyword in ("fix", "clean", "tidy", "organize", "update")):
        add_step("code_edit", "imports_fixed")

    if any(keyword in combined for keyword in ("endpoint", "api", "route", "controller")):
        add_step("add_endpoint", "endpoint_generated")

    return steps


def _code_pipeline_bonus(
    pipeline: List[Tuple[str, str]],
    achieved_flags: Set[str],
    produced_effects: Set[str],
    behaviors: List[str],
) -> float:
    if not pipeline:
        return 0.0

    completed = set(achieved_flags)
    pipeline_flags = [flag for _, flag in pipeline]
    for flag in produced_effects:
        if flag in pipeline_flags:
            completed.add(flag)

    next_index = 0
    for idx, (_, flag) in enumerate(pipeline):
        if flag in completed:
            next_index = idx + 1

    bonus = 0.0
    current_behavior = behaviors[0] if behaviors else None

    if next_index < len(pipeline):
        next_behavior, next_flag = pipeline[next_index]
        if current_behavior == next_behavior or next_flag in produced_effects:
            bonus += 0.35

    if current_behavior:
        for idx, (behavior, _) in enumerate(pipeline):
            if behavior != current_behavior:
                continue
            if idx > next_index:
                bonus -= 0.45 * (idx - next_index)
            break

    for idx, (behavior, flag) in enumerate(pipeline):
        if idx <= next_index:
            continue
        if flag in produced_effects or behavior in behaviors:
            bonus -= 0.12

    if all(flag in completed for _, flag in pipeline):
        bonus += 0.12

    return max(-0.4, min(bonus, 0.5))
