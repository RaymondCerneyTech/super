from __future__ import annotations

import copy
import heapq
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
    initial_flags: Set[str] = set()

    heap: List[Tuple[float, int, Dict[str, Any]]] = []
    counter = 0
    state = {
        "utility": 0.0,
        "ctx": initial_ctx,
        "flags": initial_flags,
        "steps": [],
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

            utility = current["utility"] + overall - 0.05 * cost - bias_penalty + progress_bonus + meaning_bonus

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
