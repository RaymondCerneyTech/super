from __future__ import annotations

import copy
import heapq
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

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
) -> Dict[str, Any]:
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
    if current_meaning == "code_edit":
        if "code_update_detected" not in goal_flags:
            goal_flags.append("code_update_detected")
        for _, flag in code_pipeline:
            if flag not in goal_flags:
                goal_flags.append(flag)
    cache_key = cache.build_key(
        goal_flags,
        backend=str(data.get("index_backend", "tfidf")),
        verbosity=str(data.get("verbosity", "normal")),
        tags=tags,
        meaning=meaning_signature,
    )
    if use_cache:
        cached_behaviors = cache.lookup(cache_key)
        if cached_behaviors:
            return _execute_cached_plan(cached_behaviors, ctx, registry, interpreter, goal_flags)

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

        for behavior in registry.list():
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

    best["goal_satisfied"] = bool(goal_set and goal_set.issubset(best["flags"])) if goal_set else True
    best["expansions"] = expansions
    best["remaining_flags"] = list(goal_set - set(best["flags"]))
    if code_pipeline:
        best["code_pipeline"] = code_pipeline
    if best.get("goal_satisfied"):
        cache.store(cache_key, best.get("steps", []))
    return best


def _execute_cached_plan(
    behaviors: List[str],
    ctx: Context,
    registry: BehaviorRegistry,
    interpreter,
    goal_flags: Iterable[str],
) -> Dict[str, Any]:
    execution_ctx = copy.deepcopy(ctx)
    steps: List[Tuple[str, Dict[str, Any]]] = []
    flags: Set[str] = set()
    for behavior in behaviors:
        try:
            result = interpreter.execute(behavior, execution_ctx)
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
        "ctx": execution_ctx,
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
    if next_index < len(pipeline):
        next_behavior, next_flag = pipeline[next_index]
        if next_flag in produced_effects or next_behavior in behaviors:
            bonus += 0.18

    for idx, (behavior, flag) in enumerate(pipeline):
        if idx <= next_index:
            continue
        if flag in produced_effects or behavior in behaviors:
            bonus -= 0.12

    if all(flag in completed for _, flag in pipeline):
        bonus += 0.1

    return max(-0.25, min(bonus, 0.35))
