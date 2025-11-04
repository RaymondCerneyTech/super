from __future__ import annotations

import copy
from typing import Callable, Dict

from core.interfaces import Context
from core.planner import normalise_goal_flags, plan
from main import build_runtime

PLANNERS: Dict[str, Callable[[Context], Dict]] = {}

TOT_BREADTH = 3
TOT_DEPTH = 2


def _prepare_context(ctx: Context) -> tuple[Context, list[str], str]:
    if not isinstance(ctx, dict):
        raise TypeError("planner context must be a dict-like Context")
    goal_text = ctx.get("goal")
    if not isinstance(goal_text, str) or not goal_text.strip():
        raise ValueError("context requires a non-empty 'goal' string")
    working_ctx: Context = copy.deepcopy(ctx)
    data_layer = working_ctx.setdefault("data", {})
    if not isinstance(data_layer, dict):
        data_layer = {}
        working_ctx["data"] = data_layer
    text = working_ctx.get("text") or data_layer.get("text") or ""
    if isinstance(text, str):
        data_layer.setdefault("text", text)
    router_state = working_ctx.setdefault("router", {})
    if not isinstance(router_state, dict):
        router_state = {}
        working_ctx["router"] = router_state
    router_state["goal_text"] = goal_text
    router_state.setdefault("no_bandit", False)
    goal_flags = normalise_goal_flags(goal_text)
    return working_ctx, goal_flags, goal_text


def _run_base_plan(ctx: Context, max_expansions: int = 20) -> Dict:
    working_ctx, goal_flags, goal_text = _prepare_context(ctx)
    registry, interpreter, router = build_runtime()
    cluster = router.cluster_hint(goal_text, working_ctx)
    working_ctx["router"]["cluster_bias"] = cluster
    result = plan(
        goal_flags,
        working_ctx,
        registry,
        interpreter,
        cluster_bias=cluster,
        max_expansions=max_expansions,
    )
    result = copy.deepcopy(result)
    result["planner"] = "first_principles"
    return result


def planner_fp(ctx: Context) -> Dict:
    return _run_base_plan(ctx, max_expansions=int(ctx.get("max_expansions", 20)))


def planner_analogy(ctx: Context) -> Dict:
    result = _run_base_plan(ctx, max_expansions=int(ctx.get("max_expansions", 16)))
    result["planner"] = "analogy"
    return result


def planner_causal(ctx: Context) -> Dict:
    result = _run_base_plan(ctx, max_expansions=int(ctx.get("max_expansions", 18)))
    result["planner"] = "causal"
    return result


def planner_react(ctx: Context) -> Dict:
    base = _run_base_plan(ctx, max_expansions=int(ctx.get("max_expansions", 12)))
    react_trace = []
    limited_steps = base.get("steps", [])[:6]
    for index, step in enumerate(limited_steps):
        behavior, info = step
        react_trace.append(
            (
                "thought",
                {
                    "index": index,
                    "text": f"Consider executing {behavior}",
                    "score": info.get("rewards", {}).get("overall", 0.0),
                },
            )
        )
        react_trace.append(
            (
                "action",
                step,
            )
        )
    base["react_trace"] = react_trace
    base["planner"] = "react"
    return base


def planner_tot(ctx: Context) -> Dict:
    base = _run_base_plan(ctx, max_expansions=int(ctx.get("max_expansions", 8)))
    steps = base.get("steps", [])
    search_tree = []
    for depth in range(1, min(TOT_DEPTH * TOT_BREADTH + 1, len(steps) + 1)):
        path = steps[:depth]
        score = 0.0
        for _, info in path:
            score += float(info.get("rewards", {}).get("overall", 0.0))
        search_tree.append({"path": path, "score": score})
        if len(search_tree) >= TOT_BREADTH * TOT_DEPTH:
            break
    base["search_tree"] = search_tree
    base["planner"] = "tree_of_thoughts"
    return base


PLANNERS.update(
    {
        "planner_fp": planner_fp,
        "planner_analogy": planner_analogy,
        "planner_causal": planner_causal,
        "planner_react": planner_react,
        "planner_tot": planner_tot,
    }
)


def run_planner(name: str, ctx: Context) -> Dict:
    try:
        planner = PLANNERS[name]
    except KeyError as exc:
        raise ValueError(f"unknown planner '{name}'") from exc
    return planner(ctx)


__all__ = [
    "PLANNERS",
    "planner_fp",
    "planner_analogy",
    "planner_causal",
    "planner_react",
    "planner_tot",
    "run_planner",
]
