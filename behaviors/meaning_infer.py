from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from core.interfaces import Behavior, Context, Result

MEANING_OPTIONS: Tuple[str, ...] = (
    "compress_to_essence",
    "transform_style",
    "ground_and_cite",
    "analyze_and_comment",
    "plan_and_execute",
    "code_edit",
)

KEYWORD_HINTS: Dict[str, Iterable[str]] = {
    "compress_to_essence": ("summarize", "summary", "brief", "tl;dr", "condense", "essence", "short"),
    "transform_style": ("rewrite", "tone", "style", "voice", "persona", "brand", "rephrase"),
    "ground_and_cite": ("cite", "citation", "source", "evidence", "grounded", "reference", "attribution"),
    "analyze_and_comment": ("analyze", "analysis", "insight", "comment", "explain why", "diagnose", "sentiment"),
    "plan_and_execute": ("plan", "steps", "task list", "execute", "automation", "workflow", "todo"),
    "code_edit": (
        "refactor",
        "add endpoint",
        "add new endpoint",
        "add a new endpoint",
        "add behavior",
        "fix import",
        "update router",
    ),
}

DATA_FLAGS: Dict[str, Iterable[str]] = {
    "compress_to_essence": ("max_words", "summary", "summaries", "concise"),
    "transform_style": ("tone", "style", "voice"),
    "ground_and_cite": ("cited", "grounded", "citations", "references"),
    "analyze_and_comment": ("analysis", "insight", "sentiment"),
    "plan_and_execute": ("task", "tasks", "steps", "commands"),
    "code_edit": ("code_update", "refactor", "endpoint"),
}


class MeaningInfer(Behavior):
    name = "meaning_infer"
    inputs: List[str] = []
    outputs: List[str] = ["meaning"]

    def run(self, ctx: Context) -> Result:
        data = ctx.setdefault("data", {})

        prior = data.get("meaning")
        text_sources = [
            ctx.get("text"),
            data.get("text"),
            data.get("task"),
            data.get("question"),
            data.get("goal"),
        ]
        combined = " ".join(str(chunk) for chunk in text_sources if isinstance(chunk, str))
        meaning = _infer_meaning(combined, data, prior)
        data["meaning"] = meaning

        return {
            "ok": True,
            "output": {"meaning": meaning},
            "effects": ["meaning_inferred"],
            "logs": [f"Inferred meaning intent: {meaning}"],
            "rationale": {
                "why": f"Detected task intent '{meaning}'",
                "evidence": [combined[:120]] if combined else [],
            },
            "rewards": {"overall": 1.0},
        }


def _infer_meaning(text: str, data: Dict[str, object], prior: object) -> str:
    text_lower = text.lower()
    scores: Dict[str, float] = {meaning: 0.0 for meaning in MEANING_OPTIONS}

    for meaning, keywords in KEYWORD_HINTS.items():
        for keyword in keywords:
            if keyword in text_lower:
                scores[meaning] += 1.5

    for meaning, keys in DATA_FLAGS.items():
        for key in keys:
            if _flag_present(key, data):
                scores[meaning] += 1.0

    if "endpoint" in text_lower and ("add" in text_lower or "create" in text_lower):
        scores["code_edit"] += 1.5
    if "router" in text_lower and "update" in text_lower:
        scores["code_edit"] += 1.5

    word_count = len(text_lower.split())
    if word_count > 0:
        if word_count <= 60:
            scores["compress_to_essence"] += 0.3
        if word_count >= 80:
            scores["analyze_and_comment"] += 0.4

    if isinstance(prior, str) and prior in scores:
        scores[prior] += 0.75

    best = max(scores.items(), key=lambda item: (item[1], -MEANING_OPTIONS.index(item[0])))[0]
    if scores[best] <= 0.0:
        return prior if isinstance(prior, str) and prior in MEANING_OPTIONS else "compress_to_essence"
    return best


def _flag_present(key: str, data: Dict[str, object]) -> bool:
    if key in data:
        value = data[key]
        if isinstance(value, (bool, int, float)):
            return bool(value)
        if isinstance(value, str):
            return bool(value.strip())
    return False


__all__ = ["MeaningInfer"]
