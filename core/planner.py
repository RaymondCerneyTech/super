
from __future__ import annotations

import heapq
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core.learn import BanditLearner
from core.rewards import aggregate_reward, ensure_reward_dict
from core.registry import BehaviorRegistry

DEFAULT_BEHAVIORS = [
    "summarize",
    "rewrite_style",
    "grammar_correction",
    "sentiment_analysis",
    "outline_generator",
    "report_from_data",
    "policy_check",
    "document_formatting",
    "social_post_optimize",
]
DEFAULT_EXPECTED_REWARD = 0.6
DEFAULT_MAX_DEPTH = 3
MAX_BRANCHING = 4


def plan_task(
    goal: str,
    registry: BehaviorRegistry,
    learner: Optional[BanditLearner] = None,
    history: Optional[Iterable[Dict[str, Any]]] = None,
    max_depth: int = DEFAULT_MAX_DEPTH,
    feedback: Optional[Dict[str, Any]] = None,
    **_: Any,
) -> List[str]:
    goal = (goal or "").strip()
    available = set(registry.list())
    if not goal or not available:
        return []

    subgoals = _parse_subgoals(goal)
    if len(subgoals) > 1:
        plan: List[str] = []
        for sub in subgoals:
            plan.extend(
                plan_task(
                    sub,
                    registry,
                    learner=learner,
                    history=history,
                    max_depth=max_depth,
                    feedback=feedback,
                )
            )
        return plan

    behavior_stats = _collect_behavior_stats(learner, history)
    lower_goal = goal.lower()
    if "summarize" in lower_goal and "rewrite" not in lower_goal:
        seq = [b for b in ("summarize",) if b in available]
        if seq:
            return seq
    if "summarize" in lower_goal and "rewrite" in lower_goal:
        sequence = [b for b in ("summarize", "rewrite_style") if b in available]
        if sequence:
            return sequence

    if any(word in lower_goal for word in ("condense", "shorten", "compress")) and "rewrite" in lower_goal:
        sequence = [b for b in ("summarize", "rewrite_style") if b in available]
        if sequence:
            return sequence

    avoided = set(feedback.get("avoid", [])) if feedback else set()
    candidates = _candidate_behaviors(goal, available)
    return _a_star_plan(goal, candidates, behavior_stats, max_depth, avoided)


def _parse_subgoals(goal: str) -> List[str]:
    lowered = goal.lower()
    if any(marker in lowered for marker in [" and ", " after ", ";", " then ", " afterwards "]):
        parts = [part.strip() for part in re.split(r"\band\b|\bafter\b|;|then|afterwards", goal) if part.strip()]
        if len(parts) > 1:
            if " after " in lowered:
                return list(reversed(parts))
            return parts
    return [goal]


def _candidate_behaviors(goal: str, available: set[str]) -> List[Tuple[str, int]]:
    goal_lower = goal.lower()
    ranking: List[Tuple[int, str]] = []
    keyword_map = [
        ("grammar_correction", ["grammar", "proofread", "style", "clean"], 3),
        ("sentiment_analysis", ["sentiment", "feel", "emotion", "angry", "joyful", "sarcasm"], 3),
        ("summarize", ["summarize", "summary", "tl;dr", "shorten", "condense"], 2),
        ("rewrite_style", ["rewrite", "tone", "professional", "casual"], 2),
        ("outline_generator", ["outline", "sections", "plan"], 2),
        ("report_from_data", ["report", "data", "dashboard", "metrics"], 2),
        ("policy_check", ["policy", "compliance", "banned", "restricted"], 3),
        ("document_formatting", ["format", "document", "layout"], 2),
        ("social_post_optimize", ["social", "post", "hashtag", "twitter", "linkedin", "instagram"], 2),
    ]

    for behavior, keywords, weight in keyword_map:
        if behavior not in available:
            continue
        score = sum(weight for keyword in keywords if keyword in goal_lower)
        if score:
            ranking.append((-score, behavior))

    for behavior in sorted(available):
        if all(b != behavior for _, b in ranking):
            ranking.append((-1, behavior))

    ranking.sort()
    top_score = -ranking[0][0] if ranking else 0
    selected = [(behavior, -score) for score, behavior in ranking[:MAX_BRANCHING]]
    if top_score > 1 and selected:
        return [selected[0]]
    return selected


def _collect_behavior_stats(
    learner: Optional[BanditLearner],
    history: Optional[Iterable[Dict[str, Any]]],
) -> Dict[str, float]:
    stats: Dict[str, List[float]] = {}

    if learner is not None:
        for feature_map in learner.values.values():
            for behavior, mean_reward in feature_map.items():
                stats.setdefault(behavior, []).append(float(mean_reward))

    if history:
        for entry in history:
            behavior = entry.get("behavior")
            if not behavior:
                continue
            reward = ensure_reward_dict(entry.get("reward"))
            stats.setdefault(behavior, []).append(aggregate_reward(reward))

    aggregated: Dict[str, float] = {}
    for behavior, values in stats.items():
        if values:
            aggregated[behavior] = sum(values) / len(values)

    return aggregated


def _expected_reward(behavior: str, stats: Dict[str, float]) -> float:
    return max(0.0, min(1.0, stats.get(behavior, DEFAULT_EXPECTED_REWARD)))


def _heuristic(remaining_depth: int, stats: Dict[str, float]) -> float:
    if not stats:
        avg = DEFAULT_EXPECTED_REWARD
    else:
        avg = sum(stats.values()) / len(stats)
    return remaining_depth * (1 - avg)


def _a_star_plan(
    goal: str,
    candidates: List[Tuple[str, int]],
    behavior_stats: Dict[str, float],
    max_depth: int,
    avoided_behaviors: Iterable[str],
) -> List[str]:
    avoided = set(avoided_behaviors)
    frontier: List[Tuple[float, Tuple[float, List[str], set[str]]]] = []
    heapq.heappush(frontier, (0.0, (0.0, [], set())))
    best_sequence: List[str] = []
    best_score = float("inf")

    while frontier:
        _, (cost_so_far, path, used) = heapq.heappop(frontier)

        if path and cost_so_far < best_score:
            best_sequence = path
            best_score = cost_so_far
            if len(path) >= max_depth:
                break

        if len(path) >= max_depth:
            continue

        ordered = sorted(candidates, key=lambda item: (-_expected_reward(item[0], behavior_stats), -item[1]))
        for behavior, match_score in ordered:
            if len(path) >= max_depth:
                break
            if path and behavior == path[-1]:
                continue

            expected = _expected_reward(behavior, behavior_stats)
            penalty = 1 - expected
            if behavior in avoided:
                penalty += 0.3
            if behavior in used:
                penalty += 0.1
            if match_score:
                penalty -= min(0.25, 0.05 * match_score)

            new_cost = cost_so_far + penalty
            new_path = path + [behavior]
            remaining = max_depth - len(new_path)
            heuristic = _heuristic(remaining, behavior_stats)
            priority = new_cost + heuristic
            heapq.heappush(frontier, (priority, (new_cost, new_path, used | {behavior})))

    if not best_sequence and candidates:
        return [behavior for behavior, _ in candidates[:max_depth]]
    return best_sequence


__all__ = ["plan_task"]
