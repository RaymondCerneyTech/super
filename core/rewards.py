from __future__ import annotations

import re
from typing import Any, Dict, Iterable


def ensure_reward_dict(value: Any) -> Dict[str, float]:
    """
    Normalize an arbitrary reward value into a dictionary of float components.
    """
    reward: Dict[str, float] = {}
    if isinstance(value, dict):
        for key, component in value.items():
            try:
                reward[str(key)] = float(component)
            except (TypeError, ValueError):
                continue
    elif value is not None:
        try:
            reward["overall"] = float(value)
        except (TypeError, ValueError):
            reward = {}

    if reward and "overall" not in reward:
        reward["overall"] = aggregate_reward(reward)
    return reward


def aggregate_reward(reward: Dict[str, float]) -> float:
    """
    Produce a scalar estimate from a reward dictionary by averaging non-overall factors.
    """
    components = [float(v) for k, v in reward.items() if k != "overall"]
    if not components:
        components = [float(v) for v in reward.values()]
    return sum(components) / len(components) if components else 0.0


def merge_rewards(base: Dict[str, float], updates: Iterable[tuple[str, float]]) -> Dict[str, float]:
    """
    Update reward components with new values and refresh the aggregate.
    """
    for key, value in updates:
        try:
            base[key] = float(value)
        except (TypeError, ValueError):
            continue
    base["overall"] = aggregate_reward(base)
    return base


def compute_explanation_scores(rationale: Dict[str, Any], output: Dict[str, Any]) -> Dict[str, float]:
    presence = 0.0
    specificity = 0.0
    alignment = 0.0

    if isinstance(rationale, dict) and rationale.get("why"):
        presence = 1.0
        evidence = rationale.get("evidence")
        if isinstance(evidence, list) and evidence:
            clean_evidence = [str(ev) for ev in evidence if ev]
            if clean_evidence:
                specificity = min(1.0, len(clean_evidence) / 3.0)
                output_text = " ".join(
                    str(value)
                    for value in output.values()
                    if isinstance(value, str)
                ).lower()
                matches = 0
                for ev in clean_evidence:
                    if ev.lower() in output_text:
                        matches += 1
                alignment = matches / len(clean_evidence)

    return {
        "explanation_presence": presence,
        "explanation_specificity": specificity,
        "explanation_alignment": alignment,
    }


def apply_explanation_bonus(reward: Dict[str, float], scores: Dict[str, float]) -> Dict[str, float]:
    reward.update(scores)
    if scores:
        avg = sum(scores.values()) / len(scores)
        reward["overall"] = max(0.0, min(1.0, reward.get("overall", 0.0) + 0.1 * avg))
    else:
        reward["overall"] = reward.get("overall", 0.0)
    return reward


def citation_coverage(answer_text: str) -> float:
    if not answer_text:
        return 0.0
    sentences = [sent.strip() for sent in re.split(r"(?<=[.!?])\s+", answer_text) if sent.strip()]
    if not sentences:
        return 0.0
    cited = sum(1 for sentence in sentences if "[" in sentence and "]" in sentence)
    return min(1.0, cited / len(sentences))


def faithfulness(answer_text: str, context_text: str) -> float:
    if not answer_text or not context_text:
        return 0.0
    answer_tokens = {token.lower() for token in re.findall(r"\w+", answer_text) if len(token) > 3}
    context_tokens = {token.lower() for token in re.findall(r"\w+", context_text) if len(token) > 3}
    if not answer_tokens:
        return 0.0
    overlap = answer_tokens & context_tokens
    return min(1.0, len(overlap) / len(answer_tokens))


def verbosity_score(answer_text: str, target_words: int) -> float:
    if target_words <= 0:
        return 1.0
    words = len(re.findall(r"\w+", answer_text))
    return max(0.0, min(1.0, words / target_words))


__all__ = [
    "ensure_reward_dict",
    "aggregate_reward",
    "merge_rewards",
    "compute_explanation_scores",
    "apply_explanation_bonus",
    "citation_coverage",
    "faithfulness",
    "verbosity_score",
]