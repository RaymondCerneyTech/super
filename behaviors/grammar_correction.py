from __future__ import annotations

import re
from typing import Dict

from core.interfaces import Behavior, Context, Result


class GrammarCorrection(Behavior):
    name = "grammar_correction"
    inputs = ["text"]
    outputs = ["corrected_text"]

    def run(self, ctx: Context) -> Result:
        data = ctx.setdefault("data", {})
        text = (data.get("text") or ctx.get("text") or "").strip()
        corrected = self._correct_text(text)
        data["corrected_text"] = corrected

        edits_made = corrected != text
        logs = ["Grammar correction applied." if edits_made else "No grammar issues detected."]
        return {
            "ok": True,
            "output": {"corrected_text": corrected},
            "logs": logs,
            "checks": {},
            "reward": 0.0,
        }

    def _correct_text(self, text: str) -> str:
        if not text:
            return ""

        def repl(match: re.Match[str]) -> str:
            return match.group(0)[0].upper() + match.group(0)[1:]

        sentences = re.split(r"(?<=[.!?])\s+", text)
        normalized = " ".join(repl(re.match(r".+", sentence.strip())) if sentence else "" for sentence in sentences)
        normalized = re.sub(r"\bi\b", "I", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized


__all__ = ["GrammarCorrection"]
