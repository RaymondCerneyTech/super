from __future__ import annotations

import re
from typing import Dict

from core.interfaces import Behavior, Context, Result


class DocumentFormatting(Behavior):
    name = "document_formatting"
    inputs = ["text"]
    outputs = ["formatted_text"]

    def run(self, ctx: Context) -> Result:
        data = ctx.setdefault("data", {})
        text = (data.get("text") or ctx.get("text") or "").strip()
        style = (data.get("format_style") or ctx.get("format_style") or "business").lower()
        wrap_width = int(data.get("wrap_width") or 80)

        formatted = self._normalize_spacing(text)
        formatted = self._normalize_headings(formatted)
        formatted = self._bulletize_lists(formatted)

        if style == "business":
            formatted = self._enforce_business_style(formatted)
        elif style == "casual":
            formatted = self._enforce_casual_style(formatted)

        formatted = self._wrap_lines(formatted, wrap_width)
        data["formatted_text"] = formatted

        logs = [f"Formatting applied using style='{style}', wrap_width={wrap_width}."]
        return {
            "ok": True,
            "output": {"formatted_text": formatted},
            "logs": logs,
            "checks": {},
            "reward": 0.0,
        }

    def _normalize_spacing(self, text: str) -> str:
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"\t", "    ", text)
        text = re.sub(r"\s+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _normalize_headings(self, text: str) -> str:
        def repl(match: re.Match[str]) -> str:
            heading = match.group(1).strip()
            return heading.upper()

        return re.sub(r"^#+\s*(.+)$", repl, text, flags=re.MULTILINE)

    def _bulletize_lists(self, text: str) -> str:
        def normalize_line(line: str) -> str:
            stripped = line.strip()
            if re.match(r"[\-*\d]+\.\s", stripped):
                return "- " + stripped.split(maxsplit=1)[-1]
            return stripped

        lines = [normalize_line(line) for line in text.splitlines()]
        return "\n".join(lines)

    def _enforce_business_style(self, text: str) -> str:
        text = re.sub(r"\bcan't\b", "cannot", text, flags=re.IGNORECASE)
        text = re.sub(r"\bwon't\b", "will not", text, flags=re.IGNORECASE)
        text = re.sub(r"\b\s+pls\b", " please", text, flags=re.IGNORECASE)
        return text

    def _enforce_casual_style(self, text: str) -> str:
        text = re.sub(r"\bdo not\b", "don't", text, flags=re.IGNORECASE)
        text = re.sub(r"\bwill not\b", "won't", text, flags=re.IGNORECASE)
        return text

    def _wrap_lines(self, text: str, width: int) -> str:
        if width <= 0:
            return text
        wrapped_lines = []
        for paragraph in text.split("\n"):
            words = paragraph.split()
            if not words:
                wrapped_lines.append("")
                continue
            current = []
            current_length = 0
            for word in words:
                if current and current_length + len(word) + 1 > width:
                    wrapped_lines.append(" ".join(current))
                    current = [word]
                    current_length = len(word)
                else:
                    if current:
                        current.append(word)
                        current_length += len(word) + 1
                    else:
                        current = [word]
                        current_length = len(word)
            if current:
                wrapped_lines.append(" ".join(current))
        return "\n".join(wrapped_lines)


__all__ = ["DocumentFormatting"]
