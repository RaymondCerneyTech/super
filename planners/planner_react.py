from __future__ import annotations

import copy
from typing import Any, Dict, List

from core.interfaces import Context
from core.planner import plan
from main import build_runtime
from planners.base import prepare_context

MAX_REACT_STEPS = 6

ACTION_TEMPLATES = {
    "web_search": "search({query})",
    "web_read": "read({url})",
    "quote": "quote({span})",
}


def _format_action(entry: Dict[str, Any]) -> str:
    name = str(entry.get("action") or "")
    args = entry.get("args") or {}
    template = ACTION_TEMPLATES.get(name)
    if template:
        formatted = template.format(
            query=args.get("query", ""),
            url=args.get("url", ""),
            span=args.get("span_id") or args.get("doc_id") or "",
        )
        return formatted
    return f"use({name}, {args})"


def run(ctx: Context) -> Dict[str, Any]:
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
        max_expansions=min(MAX_REACT_STEPS, int(ctx.get("max_expansions", MAX_REACT_STEPS))),
        max_wall_ms=max_wall_ms,
    )
    result = copy.deepcopy(result)
    react_trace = result.get("react_trace") or working_ctx.get("data", {}).get("react_trace") or []
    dialogue: List[Dict[str, Any]] = []
    for entry in react_trace[:MAX_REACT_STEPS]:
        iteration = entry.get("iteration")
        dialogue.append(
            {
                "type": "thought",
                "step": iteration,
                "text": entry.get("thought", ""),
            }
        )
        dialogue.append(
            {
                "type": "action",
                "step": iteration,
                "text": _format_action(entry),
            }
        )
        dialogue.append(
            {
                "type": "observation",
                "step": iteration,
                "text": entry.get("observation", ""),
            }
        )

    result["react_dialogue"] = dialogue
    result["planner"] = "react"
    return result


__all__ = ["run"]
