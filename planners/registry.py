from __future__ import annotations

import copy
from typing import Callable, Dict

from core.interfaces import Context
from planners.base import run_base_plan
from planners import planner_react as planner_react_module
from planners import planner_tot as planner_tot_module

PLANNERS: Dict[str, Callable[[Context], Dict]] = {}

def planner_fp(ctx: Context) -> Dict:
    result = run_base_plan(ctx, max_expansions=int(ctx.get("max_expansions", 20)))
    result["planner"] = "first_principles"
    return result


def planner_analogy(ctx: Context) -> Dict:
    result = run_base_plan(ctx, max_expansions=int(ctx.get("max_expansions", 16)))
    result["planner"] = "analogy"
    return result


def planner_causal(ctx: Context) -> Dict:
    result = run_base_plan(ctx, max_expansions=int(ctx.get("max_expansions", 18)))
    result["planner"] = "causal"
    return result


def planner_react(ctx: Context) -> Dict:
    return planner_react_module.run(ctx)


def planner_tot(ctx: Context) -> Dict:
    return planner_tot_module.run(ctx)


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
