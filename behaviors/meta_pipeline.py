from __future__ import annotations

from typing import Dict, List

from core.interfaces import Behavior, Context, Result
from core.meta_planner import build_features, choose_planners, generate_candidates


class MetaPipeline(Behavior):
    name = "meta_pipeline"
    inputs: List[str] = ["goal"]
    outputs: List[str] = ["candidates"]

    def run(self, ctx: Context) -> Result:
        data = ctx.setdefault("data", {})
        requested = max(1, int(data.get("n_candidates") or 1))
        max_planners = int(data.get("max_planners") or 0)
        max_candidates = int(data.get("max_candidates") or 0)
        judge_bundle = data.get("judge_bundle")

        features = build_features(ctx)
        planner_names = choose_planners(features, max_planners=max_planners if max_planners else None)
        if not planner_names:
            planner_names = ["planner_fp"]

        target_total = requested
        if max_candidates:
            target_total = min(target_total, max_candidates)
        if target_total < 1:
            target_total = 1

        if len(planner_names) > target_total:
            planner_names = planner_names[:target_total]

        num_planners = len(planner_names)
        k_per = max(1, min(3, target_total // num_planners if num_planners else 1))
        info = generate_candidates(
            planner_names,
            ctx,
            k_per=k_per,
            judge_bundle=judge_bundle,
            target_total=target_total,
        )
        candidates = info["candidates"]
        if not candidates:
            single = generate_candidates([planner_names[0]], ctx, k_per=1, judge_bundle=judge_bundle, target_total=1)
            candidates = single["candidates"]
            info = single

        if max_candidates:
            candidates = candidates[:max_candidates]

        data["candidates"] = candidates
        data["planner_arm"] = info.get("arm_id")
        data["planner_bundle"] = list(info.get("planner_bundle", []))
        data["judge_bundle"] = list(info.get("judge_bundle", []))
        data["planner_features"] = info.get("features", {})

        planners_used = info.get("planner_bundle", tuple()) or tuple(planner_names)
        return {
            "ok": True,
            "output": {"candidates": candidates},
            "effects": ["have_plan_candidates"],
            "logs": [f"Generated {len(candidates)} plan candidate(s) via {', '.join(planners_used)}"],
            "rewards": {"overall": 0.7},
            "rationale": {
                "why": "Meta-planner enumerated candidate plans.",
                "evidence": [planners_used[0]] if planners_used else [],
            },
        }


__all__ = ["MetaPipeline"]
