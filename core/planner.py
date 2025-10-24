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
}
DEFAULT_GOAL_BONUS = 0.25


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
    cache_key = cache.build_key(goal_flags, backend=str(data.get("index_backend", "tfidf")), verbosity=str(data.get("verbosity", "normal")), tags=tags)
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
            overall = rewards.get("overall", 0.0)
            cost = registry.behavior_cost(behavior)
            capabilities = [cap.lower() for cap in registry.behavior_capabilities(behavior)]

            bias_penalty = 0.0
            if cluster_bias == "analytic" and "creative" in capabilities:
                bias_penalty = 0.08
            elif cluster_bias == "creative":
                bias_penalty = -0.05 if "creative" in capabilities else 0.04

            new_flags = current["flags"].union(step_result.get("effects", []))
            if new_flags == current["flags"]:
                continue
            gained_flags = (new_flags - current["flags"]) & goal_set
            progress_bonus = sum(GOAL_PRIORITY_BONUS.get(flag, DEFAULT_GOAL_BONUS) for flag in gained_flags)
            utility = current["utility"] + overall - 0.05 * cost - bias_penalty + progress_bonus

            new_state = {
                "utility": utility,
                "ctx": new_ctx,
                "flags": new_flags,
                "steps": current["steps"]
                + [
                    (
                        behavior,
                        {
                            "rewards": rewards,
                            "rationale": step_result.get("rationale", {}),
                            "effects": step_result.get("effects", []),
                        },
                    )
                ],
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
