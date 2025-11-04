from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple


CALIBRATION = {"neutral": 0.5}


def _normalize_answer(candidate: Dict[str, object]) -> str:
    text = ""
    output = candidate.get("output")
    if isinstance(output, dict):
        text = output.get("final") or output.get("partial") or ""
    if isinstance(text, dict):
        text = text.get("text", "")
    if not isinstance(text, str):
        text = str(text)
    return " ".join(text.lower().split())


def rule_judge(candidate: Dict[str, object]) -> float:
    effects = candidate.get("effects", []) or []
    score = 0.0
    if "compliant" in effects:
        score += 0.2
    if "formatted" in effects:
        score += 0.2
    rewards = candidate.get("rewards", {}) or {}
    overall = float(rewards.get("overall", 0.0))
    score += 0.6 * max(0.0, min(1.0, overall))
    return max(0.0, min(1.0, score))


def consistency_judge(candidates: Iterable[Dict[str, object]]) -> Dict[int, float]:
    groups: Dict[str, List[int]] = defaultdict(list)
    idx = 0
    for idx, candidate in enumerate(candidates):
        key = _normalize_answer(candidate)
        groups[key].append(idx)
    if idx == -1:
        return {}
    votes = {i: 0.0 for i in range(idx + 1)}
    for members in groups.values():
        weight = 1.0 / len(members)
        for member in members:
            votes[member] += weight
    vote_totals = sum(votes.values()) or 1.0
    return {k: v / vote_totals for k, v in votes.items()}


def llm_pairwise_judge(a: Dict[str, object], b: Dict[str, object], rubric: str = "") -> Tuple[int, str]:
    return 0, "llm judge unavailable"


def aggregate(candidates: List[Dict[str, object]]) -> Tuple[int, Dict[str, float]]:
    if not candidates:
        raise ValueError("no candidates to judge")
    rule_scores = [rule_judge(c) for c in candidates]
    consistency_scores_dict = consistency_judge(candidates)
    consistency_scores = [consistency_scores_dict.get(i, 0.0) for i in range(len(candidates))]
    llm_scores = [0.5 for _ in candidates]
    final_scores = []
    for idx, _ in enumerate(candidates):
        score = 0.5 * rule_scores[idx] + 0.3 * consistency_scores[idx] + 0.2 * llm_scores[idx]
        final_scores.append(score)
    best_idx = max(range(len(final_scores)), key=lambda i: final_scores[i])
    return best_idx, {
        "rule": rule_scores[best_idx],
        "consistency": consistency_scores[best_idx],
        "llm": llm_scores[best_idx],
        "overall": final_scores[best_idx],
    }


__all__ = [
    "rule_judge",
    "consistency_judge",
    "llm_pairwise_judge",
    "aggregate",
]
