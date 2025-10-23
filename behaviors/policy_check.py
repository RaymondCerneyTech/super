from __future__ import annotations

import re
from typing import Dict, List

from core.interfaces import Behavior, Context, Result


class PolicyCheck(Behavior):
    name = "policy_check"
    inputs = ["text", "policies"]
    outputs = ["violations", "sanitized_text"]

    def run(self, ctx: Context) -> Result:
        data = ctx.setdefault("data", {})
        text = (data.get("text") or ctx.get("text") or "").strip()
        policies = data.get("policies") or ctx.get("policies") or []
        replacement = data.get("policy_replacement") or ctx.get("policy_replacement") or "[REDACTED]"

        violations = self._find_violations(text, policies)
        sanitized = self._apply_replacements(text, violations, replacement)

        data["violations"] = violations
        data["sanitized_text"] = sanitized

        logs = [f"Detected {len(violations)} policy violations."]
        return {
            "ok": True,
            "output": {"violations": violations, "sanitized_text": sanitized},
            "logs": logs,
            "checks": {},
            "reward": 0.0,
        }

    def _find_violations(self, text: str, policies: List[str]) -> List[Dict[str, str]]:
        violations = []
        for phrase in policies:
            if not phrase:
                continue
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            for match in pattern.finditer(text):
                violations.append({"phrase": phrase, "context": self._get_context(text, match.start(), match.end())})
        return violations

    def _get_context(self, text: str, start: int, end: int, window: int = 30) -> str:
        snippet_start = max(0, start - window)
        snippet_end = min(len(text), end + window)
        return text[snippet_start:snippet_end]

    def _apply_replacements(self, text: str, violations: List[Dict[str, str]], replacement: str) -> str:
        sanitized = text
        for violation in violations:
            phrase = violation["phrase"]
            sanitized = re.sub(re.escape(phrase), replacement, sanitized, flags=re.IGNORECASE)
        return sanitized


__all__ = ["PolicyCheck"]
