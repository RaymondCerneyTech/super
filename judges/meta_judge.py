from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from tools import llama_runner

CALIBRATION = {"neutral": 0.5}
LLM_JUDGE_MODEL_ENV = "LLM_JUDGE_MODEL"
DEFAULT_LLJ_RUBRIC = """
You are an impartial scientific judge. Evaluate which answer better satisfies the user request with accuracy,
grounding, and clarity. Consider whether key claims are supported by evidence and whether the explanation is actionable.
Respond with JSON: {"winner": "A"|"B"|"tie", "rationale": "short explanation"}.
"""


def _extract_text(candidate: Dict[str, object]) -> str:
    text = ""
    output = candidate.get("output")
    if isinstance(output, dict):
        text = output.get("final") or output.get("partial") or ""
    if isinstance(text, dict):
        text = text.get("text", "")
    if not isinstance(text, str):
        text = str(text)
    return text.strip()


def _normalize_answer(candidate: Dict[str, object]) -> str:
    return " ".join(_extract_text(candidate).lower().split())


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
    total = sum(len(members) for members in groups.values()) or 1
    for members in groups.values():
        weight = len(members) / total
        for member in members:
            votes[member] = weight
    return votes


def llm_pairwise_judge(
    a: Dict[str, object],
    b: Dict[str, object],
    rubric: str = DEFAULT_LLJ_RUBRIC,
) -> Tuple[Optional[int], str]:
    pairs = [
        ("A", 0, _extract_text(a)),
        ("B", 1, _extract_text(b)),
    ]
    random.shuffle(pairs)

    sections: List[str] = []
    label_map: Dict[str, int] = {"a": 0, "b": 1}
    for label, _, text in pairs:
        sections.append(f"[Answer {label}]\n{text or '(empty)'}")

    prompt = f"{rubric.strip()}\n\n" + "\n\n".join(sections) + "\n\nReturn a single JSON object."
    model_path = os.getenv(LLM_JUDGE_MODEL_ENV)
    if model_path:
        result = _run_llm_judge(prompt, Path(model_path))
        winner_idx, rationale = _interpret_llm_response(result, label_map)
        if winner_idx is not None:
            return winner_idx, rationale
        if rationale:
            fallback_winner = _fallback_pairwise(a, b)
            return fallback_winner, rationale or "llm_parse_failed"

    fallback = _fallback_pairwise(a, b)
    rationale = "rule_judge_fallback"
    return fallback, rationale


def _run_llm_judge(prompt: str, model_path: Path) -> Optional[Dict[str, str]]:
    try:
        return llama_runner.run_inference(
            prompt=prompt,
            model=model_path,
            n_predict=256,
            temperature=0.0,
            extra_args=["--simple-io", "--no-warmup", "-no-cnv"],
        )
    except Exception:
        return None


def _interpret_llm_response(result: Optional[Dict[str, str]], label_map: Dict[str, int]) -> Tuple[Optional[int], str]:
    if not isinstance(result, dict) or result.get("returncode") not in {"0", 0, None}:
        return None, "llm_error"
    raw_output = (result.get("stdout") or "").strip()
    if not raw_output:
        return None, "empty_llm_output"
    payload = _extract_json(raw_output)
    if not isinstance(payload, dict):
        return None, "json_parse_failed"
    winner = str(payload.get("winner", "")).strip().lower()
    rationale = str(payload.get("rationale", "")).strip()
    if winner == "tie" or not winner:
        return None, rationale or "tie"
    winner_idx = label_map.get(winner)
    return winner_idx, rationale


def _extract_json(text: str) -> Optional[Dict[str, object]]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = text[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                return None
        return None


def _fallback_pairwise(a: Dict[str, object], b: Dict[str, object]) -> Optional[int]:
    score_a = rule_judge(a)
    score_b = rule_judge(b)
    if abs(score_a - score_b) < 1e-6:
        return None
    return 0 if score_a > score_b else 1


def aggregate(candidates: List[Dict[str, object]]) -> Tuple[int, Dict[str, float]]:
    if not candidates:
        raise ValueError("no candidates to judge")
    rule_scores = [rule_judge(c) for c in candidates]
    consistency_scores_dict = consistency_judge(candidates)
    consistency_scores = [consistency_scores_dict.get(i, 0.0) for i in range(len(candidates))]

    llm_votes = [0.0 for _ in candidates]
    comparisons = 0
    llm_details: List[Dict[str, object]] = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            winner_idx, rationale = llm_pairwise_judge(candidates[i], candidates[j])
            llm_details.append({"pair": (i, j), "winner": winner_idx, "rationale": rationale})
            if winner_idx is None:
                continue
            llm_votes[winner_idx] += 1.0
            comparisons += 1

    if comparisons:
        llm_scores = [votes / comparisons for votes in llm_votes]
    else:
        llm_scores = [0.5 for _ in candidates]

    base_scores: List[float] = []
    for idx in range(len(candidates)):
        base = 0.45 * llm_scores[idx] + 0.35 * consistency_scores[idx]
        base_scores.append(base)

    best_base = max(base_scores) if base_scores else 0.0
    contenders = [i for i, score in enumerate(base_scores) if abs(score - best_base) <= 1e-6]
    if not contenders:
        contenders = list(range(len(candidates)))
    best_idx = max(contenders, key=lambda i: rule_scores[i])
    overall = 0.7 * base_scores[best_idx] + 0.3 * rule_scores[best_idx]

    return best_idx, {
        "rule": rule_scores[best_idx],
        "consistency": consistency_scores[best_idx],
        "llm": llm_scores[best_idx],
        "base": base_scores[best_idx],
        "llm_details": llm_details,
        "overall": overall,
    }


__all__ = [
    "rule_judge",
    "consistency_judge",
    "llm_pairwise_judge",
    "aggregate",
]
