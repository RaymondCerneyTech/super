from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional

from core.interfaces import Context
from core.planner import plan
from main import build_runtime
from planners.base import prepare_context

TOT_BREADTH = 3
TOT_DEPTH = 2
AGG_STRATEGIES = ["concatenate", "topic_blocks", "hybrid"]
PROFILE_CHOICES = ["research_plan", "summary_profile"]


def _score(result: Dict[str, Any]) -> float:
    steps = result.get("steps") or []
    if steps:
        _, info = steps[-1]
        rewards = info.get("rewards") or {}
        return float(rewards.get("overall") or 0.0)
    reward = result.get("reward")
    if isinstance(reward, dict):
        return float(reward.get("overall") or 0.0)
    return float(reward or 0.0)


def _execute_plan(
    context: Context,
    goal_flags: List[str],
    goal_text: str,
    *,
    max_expansions: int,
    max_wall_ms: Optional[int] = None,
) -> Dict[str, Any]:
    registry, interpreter, router = build_runtime()
    cluster = router.cluster_hint(goal_text, context)
    context["router"]["cluster_bias"] = cluster
    if max_wall_ms:
        context.setdefault("data", {}).setdefault("max_wall_ms", max_wall_ms)
    return copy.deepcopy(
        plan(
            goal_flags,
            context,
            registry,
            interpreter,
            cluster_bias=cluster,
            max_expansions=max_expansions,
            max_wall_ms=max_wall_ms,
        )
    )


def run(ctx: Context) -> Dict[str, Any]:
    working_ctx, goal_flags, goal_text = prepare_context(ctx)
    max_steps = int(ctx.get("max_expansions", 10)) or 10
    max_wall_ms = int(ctx.get("max_wall_ms") or working_ctx.get("data", {}).get("max_wall_ms") or 0)
    deadline = (time.perf_counter() + max_wall_ms / 1000.0) if max_wall_ms else None

    def _remaining_budget_ms() -> Optional[int]:
        if not deadline:
            return None
        remaining = int((deadline - time.perf_counter()) * 1000)
        return max(0, remaining)

    def _budget_spent() -> bool:
        if not deadline:
            return False
        if (deadline - time.perf_counter()) * 1000 <= 0:
            working_ctx.setdefault("data", {}).setdefault("budget_exhausted", True)
            return True
        return False

    initial_budget = _remaining_budget_ms()
    root_context = copy.deepcopy(working_ctx)
    root_result = _execute_plan(
        root_context,
        goal_flags,
        goal_text,
        max_expansions=max_steps,
        max_wall_ms=initial_budget,
    )
    best_result = root_result
    best_score = _score(root_result)
    thought_tree: List[Dict[str, Any]] = [
        {"depth": 0, "thought": "Initial reasoning", "score": best_score}
    ]

    strategies = ctx.get("tot_strategies") or AGG_STRATEGIES
    strategies = [s for s in strategies if s]
    if not strategies:
        strategies = AGG_STRATEGIES
    strategies = strategies[:TOT_BREADTH]

    profile_choices = ctx.get("tot_profiles") or PROFILE_CHOICES
    profile_choices = [p for p in profile_choices if p] or PROFILE_CHOICES

    if _budget_spent():
        return best_result

    for strat in strategies:
        if _budget_spent():
            break
        child_ctx = copy.deepcopy(working_ctx)
        child_ctx.setdefault("data", {})["aggregate_strategy"] = strat
        budget_level_one = _remaining_budget_ms()
        if budget_level_one is not None and budget_level_one <= 0:
            _budget_spent()
            break
        level_one = _execute_plan(
            child_ctx,
            goal_flags,
            goal_text,
            max_expansions=max_steps,
            max_wall_ms=budget_level_one,
        )
        level_one_score = _score(level_one)
        thought_tree.append(
            {
                "depth": 1,
                "thought": f"Try aggregation strategy '{strat}'",
                "score": level_one_score,
            }
        )
        if level_one_score > best_score:
            best_score = level_one_score
            best_result = level_one
        for profile in profile_choices[:TOT_BREADTH]:
            if _budget_spent():
                break
            grand_ctx = copy.deepcopy(child_ctx)
            grand_ctx.setdefault("data", {})["llama_profile"] = profile
            budget_level_two = _remaining_budget_ms()
            if budget_level_two is not None and budget_level_two <= 0:
                _budget_spent()
                break
            level_two = _execute_plan(
                grand_ctx,
                goal_flags,
                goal_text,
                max_expansions=max_steps,
                max_wall_ms=budget_level_two,
            )
            level_two_score = _score(level_two)
            thought_tree.append(
                {
                    "depth": 2,
                    "thought": f"Use profile '{profile}' after '{strat}'",
                    "score": level_two_score,
                }
            )
            if level_two_score > best_score:
                best_score = level_two_score
                best_result = level_two
        if len(thought_tree) >= TOT_BREADTH * TOT_DEPTH + 1:
            break

    best_result["thought_tree"] = thought_tree
    best_result["planner"] = "tree_of_thoughts"
    best_result["tot_best_score"] = best_score
    return best_result


__all__ = ["run"]
