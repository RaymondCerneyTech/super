from __future__ import annotations

from typing import Dict, List

PIPELINES: Dict[str, List[str]] = {
    "summarize": ["summarize_bullets", "numbers_guard"],
    "policy": ["policy_check_lite", "numbers_guard"],
    "numbers": ["numbers_guard"],
    "style": ["rewrite_style", "numbers_guard"],
    "plan": ["extract_actions", "causal_dot_builder"],
    "compose": ["rewrite_clarity", "numbers_guard", "policy_check_lite"],
    "web": ["web_get", "html_to_text", "extract_facts", "numbers_guard"],
    "data": ["table_detect", "csv_summary"],
}

DEFAULT_CUE = "compose"


def select_cue(text: str) -> str:
    lowered = (text or "").lower()
    if any(keyword in lowered for keyword in ("summarize", "summary", "tl;dr", "bullet")):
        return "summarize"
    if any(keyword in lowered for keyword in ("policy", "secret", "pii", "redact", "compliance")):
        return "policy"
    if any(keyword in lowered for keyword in ("number", "percent", "bps", "%", "ratio", "figure")):
        return "numbers"
    if any(keyword in lowered for keyword in ("tone", "style", "formal", "friendly", "voice")):
        return "style"
    if any(keyword in lowered for keyword in ("step", "plan", "todo", "task", "action")):
        return "plan"
    if any(keyword in lowered for keyword in ("test", "pytest", "failing", "traceback")):
        return "plan"
    return DEFAULT_CUE


__all__ = ["PIPELINES", "select_cue"]
