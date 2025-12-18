from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

from core.interfaces import Behavior, Context, Result


def _norm(value: str) -> str:
    """Normalize text for caseless comparison."""
    return unicodedata.normalize("NFKC", value).casefold()


class PolicyCheck(Behavior):
    name = "policy_check"
    inputs = ["text"]
    outputs = ["violations", "sanitized_text"]

    _DEFAULT_POLICIES = ["password", "ssn", "confidential"]

    def run(self, ctx: Context) -> Result:
        data = ctx.setdefault("data", {})
        raw_text = data.get("text") or ctx.get("text") or ""
        if not isinstance(raw_text, str):
            raw_text = str(raw_text)
        replacement = data.get("policy_replacement") or ctx.get("policy_replacement") or "[REDACTED]"

        normalized_text, index_map = self._normalize_with_mapping(raw_text)
        policies = self._prepare_policies(data.get("policies") or ctx.get("policies"))
        data["policies"] = policies

        patterns = self._compile_patterns(policies)
        hits, violations, spans = self._detect_violations(normalized_text, index_map, raw_text, patterns)
        sanitized = self._apply_replacements(raw_text, spans, replacement)

        data["violations"] = violations
        data["sanitized_text"] = sanitized

        evidence = list(dict.fromkeys(hits)) if hits else ["clean"]
        rationale = {
            "why": "Flagged and redacted prohibited phrases" if hits else "No policy violations detected",
            "evidence": evidence,
        }
        effects = ["compliant", "cited"]

        presence = 1.0
        specificity = min(1.0, len(evidence) / 3.0) if evidence else 0.0
        if not hits and "clean" in evidence:
            alignment = 1.0
        else:
            aligned = sum(1 for _, pattern in patterns if pattern.search(normalized_text))
            alignment = aligned / max(1, len(hits)) if hits else 0.0
        rewards = {
            "explanation_presence": presence,
            "explanation_specificity": specificity,
            "explanation_alignment": alignment,
        }

        logs = [f"Detected {len(violations)} policy violations."]
        output = {
            "sanitized_text": sanitized,
            "policy_evidence": " ".join(evidence),
            "violations": violations,
        }
        return {
            "ok": True,
            "output": output,
            "logs": logs,
            "checks": {},
            "reward": rewards,
            "rewards": rewards,
            "rationale": rationale,
            "effects": effects,
        }

    def _prepare_policies(self, raw_policies) -> List[str]:
        if not raw_policies:
            return list(self._DEFAULT_POLICIES)
        if isinstance(raw_policies, str):
            cleaned = raw_policies.strip()
            return [cleaned] if cleaned else list(self._DEFAULT_POLICIES)
        if isinstance(raw_policies, Sequence):
            collected = []
            for item in raw_policies:
                text = str(item).strip()
                if text:
                    collected.append(text)
            return collected or list(self._DEFAULT_POLICIES)
        return list(self._DEFAULT_POLICIES)

    def _compile_patterns(self, policies: List[str]) -> List[Tuple[str, re.Pattern[str]]]:
        compiled: List[Tuple[str, re.Pattern[str]]] = []
        for phrase in policies:
            normalized = _norm(phrase)
            if not normalized:
                continue
            if re.search(r"\W", normalized) or " " in normalized:
                pattern = re.compile(re.escape(normalized), re.IGNORECASE)
            else:
                pattern = re.compile(r"\b" + re.escape(normalized) + r"\b", re.IGNORECASE)
            compiled.append((phrase, pattern))
        return compiled

    def _detect_violations(
        self,
        normalized_text: str,
        index_map: List[int],
        raw_text: str,
        patterns: List[Tuple[str, re.Pattern[str]]],
    ) -> Tuple[List[str], List[Dict[str, str]], List[Tuple[int, int]]]:
        hits: List[str] = []
        violations: List[Dict[str, str]] = []
        spans: List[Tuple[int, int]] = []

        for phrase, pattern in patterns:
            matches = list(pattern.finditer(normalized_text))
            if not matches:
                continue
            hits.append(phrase)
            for match in matches:
                span = self._map_span(index_map, match.span())
                if span is None:
                    continue
                raw_start, raw_end = span
                spans.append(span)
                violations.append(
                    {
                        "phrase": phrase,
                        "context": self._get_context(raw_text, raw_start, raw_end),
                    }
                )
        return hits, violations, spans

    def _apply_replacements(self, text: str, spans: List[Tuple[int, int]], replacement: str) -> str:
        if not spans or replacement is None:
            return text

        merged: List[List[int]] = []
        for start, end in sorted(spans):
            if not merged:
                merged.append([start, end])
                continue
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1][1] = max(last_end, end)
            else:
                merged.append([start, end])

        pieces: List[str] = []
        cursor = 0
        for start, end in merged:
            pieces.append(text[cursor:start])
            pieces.append(replacement)
            cursor = end
        pieces.append(text[cursor:])
        return "".join(pieces)

    def _normalize_with_mapping(self, text: str) -> Tuple[str, List[int]]:
        normalized_chunks: List[str] = []
        index_map: List[int] = []
        for idx, char in enumerate(text):
            chunk = _norm(char)
            normalized_chunks.append(chunk)
            for _ in chunk:
                index_map.append(idx)
        normalized_text = "".join(normalized_chunks)
        return normalized_text, index_map

    def _map_span(self, index_map: List[int], span: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        start, end = span
        if not index_map or start >= len(index_map) or end == 0:
            return None
        start_idx = index_map[start]
        end_idx = index_map[min(end - 1, len(index_map) - 1)] + 1
        return start_idx, end_idx

    def _get_context(self, text: str, start: int, end: int, window: int = 30) -> str:
        snippet_start = max(0, start - window)
        snippet_end = min(len(text), end + window)
        return text[snippet_start:snippet_end]


__all__ = ["PolicyCheck"]
