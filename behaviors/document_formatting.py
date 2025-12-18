from __future__ import annotations

import re
from typing import Any, Dict, List, Set

from core.interfaces import Behavior, Context, Result
from judges.meta_judge import rule_judge


class DocumentFormatting(Behavior):
    name = "document_formatting"
    inputs = ["text"]
    outputs = ["formatted_text"]

    def run(self, ctx: Context) -> Result:
        data = ctx.setdefault("data", {})
        text = (
            data.get("answer")
            or data.get("aggregated_text")
            or data.get("text")
            or ctx.get("text")
            or ""
        )
        text = text.strip()
        if not text:
            return self._refusal("document_formatting: no text or answer to format", data)
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
        formatted = self._append_analysis(formatted, data)
        requires_grounding = self._requires_grounding(data)
        requires_llm = self._requires_llm_output(data)
        if requires_llm and not data.get("_llm_generated") and not data.get("skip_llama"):
            return self._refusal("LLM output required but no generation was detected.", data)
        if requires_grounding:
            grounded_ok, grounded_reason = self._check_grounding_requirements(formatted, data)
            if not grounded_ok:
                return self._refusal(grounded_reason, data)

        data["formatted_text"] = formatted
        data.setdefault("text", formatted)

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
        candidate = {
            "effects": ["formatted"] + (["llm_output"] if data.get("_llm_generated") else []),
            "rewards": {"overall": 1.0},
            "output": {"final": formatted},
        }
        rule_score = rule_judge(candidate)
        logs.append(f"rule_judge_score={rule_score:.3f}")
        if rule_score < 0.5:
            return self._refusal("Final rule judge score too low to release an answer.", data)

        data["final_summary"] = formatted
        return {
            "ok": True,
            "output": {"formatted_text": formatted},
            "logs": logs,
            "checks": {},
            "reward": {"overall": 1.0, "rule_score": rule_score},
            "rationale": rationale,
            "effects": ["formatted", "llm_output"] if data.get("_llm_generated") else ["formatted"],
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


    def _requires_grounding(self, data: Dict[str, object]) -> bool:
        flags = data.get("goal_flags")
        if isinstance(flags, str):
            flags = [frag.strip() for frag in flags.split(",")]
        if isinstance(flags, (list, tuple, set)):
            normalized = {str(flag).strip().lower() for flag in flags}
            return any(flag in {"grounded", "cited"} for flag in normalized)
        return False

    def _requires_llm_output(self, data: Dict[str, object]) -> bool:
        flags = data.get("goal_flags")
        if isinstance(flags, str):
            flags = [frag.strip() for frag in flags.split(",")]
        if isinstance(flags, (list, tuple, set)):
            normalized = {str(flag).strip().lower() for flag in flags}
            return "llm_output" in normalized
        return False

    def _check_grounding_requirements(self, formatted: str, data: Dict[str, Any]) -> tuple[bool, str]:
        sources = data.get("sources") or []
        if not isinstance(sources, list):
            sources = []
        domains = {self._domain_from_source(src) for src in sources if src.get("url")}
        if len(domains) < 2 or not data.get("triangulation_met"):
            return False, "Insufficient independent domains to support a grounded answer."
        bullet_map = self._bullet_citations(formatted)
        if not bullet_map:
            return False, "Grounded answers must include cited bullet points."
        if len(bullet_map) < 3:
            return False, "Provide at least three cited bullets before finalizing."
        citable_markers: Set[str] = {str(src.get("marker") or "").strip() for src in sources if src.get("url")}
        for bullet, markers in bullet_map.items():
            unique_markers = {marker for marker in markers if marker}
            if len(unique_markers) < 2:
                return False, f"Bullet '{bullet[:40]}...' needs citations from at least two sources."
            if not unique_markers.issubset(citable_markers):
                return False, f"Bullet '{bullet[:40]}...' references unknown sources."
        return True, ""

    def _bullet_citations(self, text: str) -> Dict[str, List[str]]:
        bullets: Dict[str, List[str]] = {}
        bullet_re = re.compile(r"^\s*(?:[-*]|\d+\.)\s+(.*)$")
        cite_re = re.compile(r"\[(S\d+)\]")
        for line in text.splitlines():
            match = bullet_re.match(line)
            if not match:
                continue
            body = match.group(1).strip()
            markers = cite_re.findall(body)
            if markers:
                bullets[body] = markers
        return bullets

    def _domain_from_source(self, src: Dict[str, Any]) -> str:
        url = str(src.get("url") or src.get("canonical_url") or "")
        if not url:
            return ""
        match = re.match(r"https?://([^/]+)", url)
        if not match:
            return ""
        host = match.group(1).lower()
        if host.startswith("www."):
            host = host[4:]
        return host

    def _append_analysis(self, text: str, data: Dict[str, Any]) -> str:
        aggregated = str(data.get("aggregated_text") or data.get("answer") or "")
        if not aggregated.strip():
            return text
        sentences = [sent.strip() for sent in re.split(r"(?<=[.!?])\s+", aggregated) if sent.strip()]
        selected: List[str] = []
        for sentence in sentences:
            lowered = sentence.lower()
            if any(keyword in lowered for keyword in ("obstacle", "challenge", "limitation")):
                selected.append(f"- Obstacles: {sentence}")
            elif any(keyword in lowered for keyword in ("resource", "collaboration", "support")):
                selected.append(f"- Resources: {sentence}")
            if len(selected) >= 3:
                break
        if not selected:
            return text
        analysis_block = "\n\n### Analysis Notes\n" + "\n".join(selected)
        return f"{text.rstrip()}{analysis_block}"

    def _refusal(self, message: str, data: Dict[str, Any]) -> Result:
        if isinstance(data, dict):
            unmet = data.setdefault("unmet_requirements", [])
            if isinstance(unmet, list):
                unmet.append(message)
        return {
            "ok": False,
            "output": {},
            "logs": [message],
            "checks": {},
            "reward": 0.0,
            "rationale": {"why": message, "evidence": []},
            "effects": [],
        }
        return {
            "ok": False,
            "output": {},
            "logs": [message],
            "checks": {},
            "reward": 0.0,
            "rationale": {"why": message, "evidence": []},
            "effects": [],
        }


__all__ = ["DocumentFormatting"]
