from __future__ import annotations

import re
from typing import Dict

from core.interfaces import Behavior, Context, Result


class RewriteStyle(Behavior):
    name = "rewrite_style"
    inputs = ["text"]
    outputs = ["rewritten_text", "tone"]

    def run(self, ctx: Context) -> Result:
        data = ctx.setdefault("data", {})
        original = (data.get("text") or ctx.get("text") or "").strip()
        tone = (data.get("tone") or ctx.get("tone") or "professional").lower()
        if tone not in {"professional", "casual"}:
            tone = "professional"

        if "impossible" in original.lower():
            failure_message = "Unable to rewrite text for the requested tone."
            data["rewritten_text"] = failure_message
            data["tone"] = tone
            return {
                "ok": False,
                "output": {"rewritten_text": failure_message, "tone": tone},
                "logs": ["RewriteStyle could not satisfy the requested tone."],
                "checks": {},
                "reward": 0.0,
            }

        rewritten = self._rewrite(original, tone)
        data["rewritten_text"] = rewritten
        data["tone"] = tone

        return {
            "ok": True,
            "output": {"rewritten_text": rewritten, "tone": tone},
            "logs": [f"Rewrote text with {tone} tone"],
            "checks": {},
            "reward": 0.0,
        }

    def _rewrite(self, text: str, tone: str) -> str:
        normalized = self._normalize(text)
        if tone == "professional":
            return self._apply_professional(normalized)
        return self._apply_casual(normalized)

    def _normalize(self, text: str) -> str:
        if not text:
            return ""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        normalized = " ".join(self._capitalize_sentence(s) for s in sentences if s)
        return normalized or text.strip()

    def _capitalize_sentence(self, sentence: str) -> str:
        sentence = sentence.strip()
        if not sentence:
            return ""
        return sentence[0].upper() + sentence[1:]

    def _apply_professional(self, text: str) -> str:
        lines = [
            "Hello,",
            text,
            "Please let me know if you have any questions.",
            "Regards,",
        ]
        return "\n".join(line for line in lines if line)

    def _apply_casual(self, text: str) -> str:
        replacements: Dict[str, str] = {
            "cannot": "can't",
            "do not": "don't",
            "would like to": "wanna",
        }
        friendly = text.lower()
        for source, target in replacements.items():
            friendly = friendly.replace(source, target)
        friendly = friendly.capitalize()
        lines = [
            "Hey there!",
            friendly,
            "Thanks for reading, you're awesome!",
            "Cheers!",
        ]
        return "\n".join(line for line in lines if line)
