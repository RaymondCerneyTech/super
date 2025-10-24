from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Tuple

_TOKEN_RE = re.compile(r"\b\w+\b")


def _tokenize(text: str) -> Iterable[str]:
    return _TOKEN_RE.findall(text.lower())


def _extract_primary_text(output: Dict[str, Any]) -> str:
    for key, value in output.items():
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, dict):
            nested = _extract_primary_text(value)
            if nested:
                return nested
    return ""


def fluency_score(text: str) -> float:
    if not text:
        return 0.0
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        sentences = [text.strip()]
    penalty = 0.0
    for sentence in sentences:
        if sentence and sentence[0].islower():
            penalty += 0.1
        if re.search(r"\s{2,}", sentence):
            penalty += 0.05
        if re.search(r"[!?]{3,}", sentence):
            penalty += 0.05
    penalty += min(0.3, len(re.findall(r",,", text)) * 0.05)
    return max(0.0, min(1.0, 1.0 - penalty))


def creativity_score(source: str, generated: str) -> float:
    if not generated:
        return 0.0
    source_tokens = list(_tokenize(source))
    generated_tokens = list(_tokenize(generated))
    if len(generated_tokens) < 2:
        return 0.0
    source_bigrams = {tuple(source_tokens[i : i + 2]) for i in range(len(source_tokens) - 1)}
    generated_bigrams = {tuple(generated_tokens[i : i + 2]) for i in range(len(generated_tokens) - 1)}
    if not generated_bigrams:
        return 0.0
    novel = len([bg for bg in generated_bigrams if bg not in source_bigrams])
    return min(1.0, novel / max(1, len(generated_bigrams)))


def relevance_score(goal: str, source: str, generated: str) -> float:
    reference = goal or source
    if not reference or not generated:
        return 0.0
    ref_tokens = set(_tokenize(reference))
    gen_tokens = set(_tokenize(generated))
    if not gen_tokens:
        return 0.0
    overlap = len(ref_tokens & gen_tokens)
    return min(1.0, overlap / len(gen_tokens))


def compute_reward_factors(
    behavior: str,
    ctx: Dict[str, Any],
    output: Dict[str, Any],
) -> Dict[str, float]:
    generated_text = _extract_primary_text(output)
    if not generated_text:
        return {}

    data = ctx.get("data", {}) if isinstance(ctx, dict) else {}
    source_text = data.get("text") if isinstance(data, dict) else None
    goal_text = ctx.get("goal") if isinstance(ctx, dict) else None

    factors = {
        "fluency": fluency_score(generated_text),
        "creativity": creativity_score(source_text or "", generated_text),
        "relevance": relevance_score(goal_text or "", source_text or "", generated_text),
    }

    return factors


__all__ = [
    "compute_reward_factors",
    "creativity_score",
    "fluency_score",
    "relevance_score",
]