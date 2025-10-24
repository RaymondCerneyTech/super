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
        formatted, heading_fixes = self._normalize_headings(formatted)
        formatted, bullet_fixes = self._bulletize_lists(formatted)

        if style == "business":
            formatted, tone_fixes = self._enforce_business_style(formatted)
        elif style == "casual":
            formatted, tone_fixes = self._enforce_casual_style(formatted)
        else:
            tone_fixes = []

        formatted = self._wrap_lines(formatted, wrap_width)
        data["formatted_text"] = formatted

        evidence = []
        if heading_fixes:
            evidence.append(f"headings: {heading_fixes}")
        if bullet_fixes:
            evidence.append(f"lists: {bullet_fixes}")
        if tone_fixes:
            evidence.append(f"tone: {tone_fixes}")
        evidence.append(f"wrap={wrap_width}")

        rationale = {
            "why": f"Formatted document in {style} style",
            "evidence": evidence,
        }

        logs = [f"Formatting applied using style='{style}', wrap_width={wrap_width}."]
        return {
            "ok": True,
            "output": {"formatted_text": formatted},
            "logs": logs,
            "checks": {},
            "reward": 0.0,
            "rationale": rationale,
            "effects": ["formatted"],
        }

    def _normalize_spacing(self, text: str) -> str:
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"\t", "    ", text)
        text = re.sub(r"\s+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _normalize_headings(self, text: str) -> tuple[str, str]:
        fixes = []

        def repl(match: re.Match[str]) -> str:
            heading = match.group(1).strip()
            fixes.append(heading)
            return heading.upper()

        return re.sub(r"^#+\s*(.+)$", repl, text, flags=re.MULTILINE), ", ".join(fixes)

    def _bulletize_lists(self, text: str) -> tuple[str, str]:
        converted = []
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if re.match(r"[\-*\d]+\.\s", stripped):
                normalized = "- " + stripped.split(maxsplit=1)[-1]
                converted.append(normalized)
                lines.append(normalized)
            else:
                lines.append(stripped)
        return "\n".join(lines), ", ".join(converted)

    def _enforce_business_style(self, text: str) -> tuple[str, list[str]]:
        fixes = []
        def replace(pattern: str, repl: str, label: str, source: str) -> str:
            nonlocal fixes
            if re.search(pattern, source, flags=re.IGNORECASE):
                fixes.append(label)
            return re.sub(pattern, repl, source, flags=re.IGNORECASE)

        text = replace(r"\bcan't\b", "cannot", "cant->cannot", text)
        text = replace(r"\bwon't\b", "will not", "wont->will not", text)
        text = replace(r"\bpls\b", "please", "pls->please", text)
        return text, fixes

    def _enforce_casual_style(self, text: str) -> tuple[str, list[str]]:
        fixes = []
        def replace(pattern: str, repl: str, label: str, source: str) -> str:
            nonlocal fixes
            if re.search(pattern, source, flags=re.IGNORECASE):
                fixes.append(label)
            return re.sub(pattern, repl, source, flags=re.IGNORECASE)

        text = replace(r"\bdo not\b", "don't", "do not->don't", text)
        text = replace(r"\bwill not\b", "won't", "will not->won't", text)
        return text, fixes

    def _wrap_lines(self, text: str, width: int) -> str:
        if width <= 0:
            return text
        wrapped_lines = []
        for paragraph in text.split("\n"):
            words = paragraph.split()
            if not words:
                wrapped_lines.append("")
                continue
            current: list[str] = []
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
