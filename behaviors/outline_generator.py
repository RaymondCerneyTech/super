from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List

from core.interfaces import Behavior, Context, Result

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "about",
    "into",
    "using",
    "this",
    "that",
    "from",
    "into",
    "your",
    "their",
    "these",
    "those",
    "what",
    "when",
    "where",
    "which",
    "will",
    "help",
}


class OutlineGenerator(Behavior):
    name = "outline_generator"
    inputs = ["topic"]
    outputs = ["outline"]

    def run(self, ctx: Context) -> Result:
        data = ctx.setdefault("data", {})
        topic = (data.get("topic") or ctx.get("topic") or "").strip()
        max_sections = int(data.get("max_sections") or 5)
        focus_keywords = data.get("focus_keywords") or []

        outline = self._build_outline(topic, max_sections, focus_keywords)
        data["outline"] = outline

        logs = [f"Generated outline with {len(outline)} sections"]
        if focus_keywords:
            logs.append(f"Focus keywords: {', '.join(focus_keywords)}")

        return {
            "ok": True,
            "output": {"outline": outline},
            "logs": logs,
            "checks": {},
            "reward": 0.0,
        }

    def _build_outline(self, topic: str, max_sections: int, focus_keywords: List[str]) -> List[str]:
        if not topic:
            return []
        tokens = self._tokenize(topic)
        keyword_scores: Dict[str, float] = Counter(tokens)
        for keyword in focus_keywords:
            keyword_scores[keyword.lower()] += 2.0

        ranked = [word for word, _ in keyword_scores.most_common(max_sections * 2) if len(word) > 2]
        seen = set()
        sections = []
        for word in ranked:
            if word in seen:
                continue
            title = word.title()
            sections.append(f"{title}: Key Considerations")
            seen.add(word)
            if len(sections) >= max_sections:
                break

        if not sections:
            sections = ["Introduction", "Key Points", "Next Steps"]
        return sections

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r"\w+", text.lower())
        return [t for t in tokens if t not in STOPWORDS]


__all__ = ["OutlineGenerator"]
