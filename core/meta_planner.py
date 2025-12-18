from __future__ import annotations

import copy
from typing import Dict, Iterable, List, Tuple

import copy

from core.bandit import UCB1
from core.credit_ledger import record as ledger_record
from core.cues import select_cue
from planners.registry import PLANNERS, run_planner

BANDITS: Dict[str, UCB1] = {}
DEFAULT_JUDGES = ("meta_judge",)


def _ensure_tuple(value) -> Tuple[str, ...]:
    if not value:
        return tuple()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _compose_arm(planner_bundle: Iterable[str], judge_bundle: Iterable[str], cue: str) -> str:
    planner_str = "&".join(planner_bundle)
    judge_str = "&".join(judge_bundle)
    return f"{planner_str}|{judge_str}|{cue}"


def choose_planners(features: Dict[str, object], max_planners: int | None = None) -> List[str]:
    cue = str(features.get("cue") or "").lower()
    length = int(features.get("text_len") or 0)
    meaning = str(features.get("meaning") or "").lower()
    goal_text = str(features.get("goal_text") or "").lower()
    grounded_goal = any(token in goal_text for token in ("grounded", "cited"))

    picks: List[str] = []
    if cue in {"summarize", "compose"} or length < 600:
        picks.extend(["planner_fp", "planner_react"])
    if cue in {"plan", "policy"} or meaning in {"plan_and_execute", "analyze_and_comment"}:
        picks.extend(["planner_tot", "planner_causal"])
    if cue == "numbers":
        picks.extend(["planner_causal", "planner_fp"])

    if not picks:
        picks.extend(["planner_fp", "planner_analogy", "planner_tot"])

    unique: List[str] = []
    for name in picks:
        if name in PLANNERS and name not in unique:
            unique.append(name)
    if not unique:
        unique = ["planner_fp"]
    if grounded_goal:
        if "planner_react" not in unique and "planner_react" in PLANNERS:
            unique.insert(0, "planner_react")
        if "planner_tot" in PLANNERS and "planner_tot" not in unique:
            unique.append("planner_tot")

    if max_planners is not None and max_planners > 0:
        unique = unique[:max_planners]

    bandit = BANDITS.get(cue)
    if bandit and bandit.has_arms():
        preferred = bandit.pick()
        if preferred:
            parts = preferred.split("|")
            if len(parts) == 3 and parts[2] == cue:
                planner_str = parts[0]
                preferred_planners = [p for p in planner_str.split("&") if p]
                ordered = preferred_planners + [p for p in unique if p not in preferred_planners]
                unique = [p for p in ordered if p in unique]

    return unique


def generate_candidates(
    names: Iterable[str],
    ctx: Dict[str, object],
    *,
    k_per: int = 1,
    judge_bundle: Iterable[str] | None = None,
    target_total: int | None = None,
) -> Dict[str, object]:
    names = list(names)
    if not names:
        names = ["planner_fp"]
    planner_bundle = tuple(dict.fromkeys(names))

    data_layer = ctx.get("data") if isinstance(ctx, dict) else {}
    if judge_bundle is None and isinstance(data_layer, dict):
        judge_bundle = data_layer.get("judge_bundle")
    judge_bundle = _ensure_tuple(judge_bundle) or DEFAULT_JUDGES

    features = build_features(ctx)
    cue = str(features.get("cue") or "generic")
    arm_id = _compose_arm(planner_bundle, judge_bundle, cue)

    bandit = BANDITS.setdefault(cue, UCB1())
    bandit.ensure_arm(arm_id)

    candidates: List[Dict[str, object]] = []
    total_limit = target_total if (target_total and target_total > 0) else None

    goal_text = ""
    if isinstance(ctx, dict):
        goal_text = str(ctx.get("goal") or "")
        data_ref = ctx.get("data")
        if isinstance(data_ref, dict):
            goal_text = str(data_ref.get("goal") or goal_text)
    grounded_goal = any(token in goal_text.lower() for token in ("grounded", "cited"))

    for name in planner_bundle:
        runs = max(1, min(3, k_per))
        for _ in range(runs):
            if total_limit is not None and len(candidates) >= total_limit:
                break
            working_ctx = copy.deepcopy(ctx)
            try:
                result = run_planner(name, working_ctx)
            except Exception as exc:  # pragma: no cover - defensive
                result = {"planner": name, "error": str(exc), "steps": []}
            result["planner_name"] = name
            candidates.append(result)
            if (
                name == "planner_react"
                and grounded_goal
                and not result.get("goal_satisfied", False)
                and "planner_tot" in PLANNERS
                and not any(c.get("planner_name") == "planner_tot" for c in candidates)
            ):
                tot_ctx = copy.deepcopy(ctx)
                tot_result = run_planner("planner_tot", tot_ctx)
                tot_result["planner_name"] = "planner_tot"
                candidates.append(tot_result)
        if total_limit is not None and len(candidates) >= total_limit:
            break

    if total_limit is not None:
        candidates = candidates[:total_limit]

    return {
        "candidates": candidates,
        "arm_id": arm_id,
        "planner_bundle": planner_bundle,
        "judge_bundle": judge_bundle,
        "features": features,
    }


def record_outcome(arm_id: str, features: Dict[str, object], reward: float, metadata: Dict[str, object] | None = None) -> None:
    if not arm_id:
        return
    cue = str(features.get("cue") or "generic")
    bandit = BANDITS.setdefault(cue, UCB1())
    bandit.ensure_arm(arm_id)
    bandit.record(arm_id, reward)
    payload = metadata.copy() if isinstance(metadata, dict) else {}
    payload.setdefault("cue", cue)
    ledger_record(
        call_site="meta_planner",
        tool=arm_id,
        score={"ok": reward >= 1.0, "delta_quality": reward, "faithfulness": payload.get("rule_score", 0.0)},
        meta=payload,
    )


def build_features(ctx: Dict[str, object]) -> Dict[str, object]:
    data = ctx.get("data") if isinstance(ctx, dict) else {}
    text = ""
    if isinstance(ctx, dict):
        text = ctx.get("text") or ""
    if isinstance(data, dict):
        text = data.get("text") or text
    cue = select_cue(str(text))
    meaning = ""
    if isinstance(data, dict):
        meaning = str(data.get("meaning") or "")
    goal_text = ""
    if isinstance(ctx, dict):
        goal_text = str(ctx.get("goal") or "")
    if isinstance(data, dict):
        goal_text = str(data.get("goal") or goal_text)
    return {
        "cue": cue,
        "meaning": meaning,
        "text_len": len(text or ""),
        "goal_text": goal_text,
    }


__all__ = [
    "choose_planners",
    "generate_candidates",
    "build_features",
    "record_outcome",
]
