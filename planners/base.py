from __future__ import annotations

import copy
from typing import Dict, List, Tuple

from core.interfaces import Context
from core.planner import normalise_goal_flags, plan
from main import build_runtime


def prepare_context(ctx: Context) -> Tuple[Context, List[str], str]:
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


def run_base_plan(ctx: Context, max_expansions: int = 20) -> Dict:
    working_ctx, goal_flags, goal_text = prepare_context(ctx)
    registry, interpreter, router = build_runtime()
    cluster = router.cluster_hint(goal_text, working_ctx)
    working_ctx["router"]["cluster_bias"] = cluster
    raw_wall = ctx.get("max_wall_ms") or working_ctx.get("data", {}).get("max_wall_ms")
    max_wall_ms = int(raw_wall) if raw_wall else None
    result = plan(
        goal_flags,
        working_ctx,
        registry,
        interpreter,
        cluster_bias=cluster,
        max_expansions=max_expansions,
        max_wall_ms=max_wall_ms,
    )
    return copy.deepcopy(result)


__all__ = ["prepare_context", "run_base_plan"]
