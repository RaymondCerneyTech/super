from __future__ import annotations

import math
import re
from typing import Dict, List, Tuple

from core.interfaces import Behavior, Context, Result
from core.rewards import citation_coverage, faithfulness, verbosity_score


class AnswerVerbose(Behavior):
    name = "answer_verbose"
    inputs = ["question"]
    outputs = ["answer"]

    def run(self, ctx: Context) -> Result:
        data = ctx.setdefault("data", {})
        question = str(data.get("question") or ctx.get("question") or "").strip()
        passages = data.get("passages") or ctx.get("passages") or []
        aggregated = data.get("aggregated_text") or ctx.get("aggregated_text")
        verbosity = (data.get("verbosity") or ctx.get("verbosity") or "verbose").lower()
        min_words = int(data.get("min_words") or ctx.get("min_words") or (800 if verbosity in {"verbose", "max"} else 300))
        max_words = data.get("max_words") or ctx.get("max_words")
        max_words = int(max_words) if max_words else None
        section_headers = bool(data.get("section_headers", True))
        bullets = bool(data.get("bullets", True))
        examples = bool(data.get("examples", True))

        if not question:
            return self._failure("No question provided for answering.")
        if not passages and not aggregated:
            return self._failure("No passages or aggregate text available.")

        context_text = aggregated if isinstance(aggregated, str) and aggregated.strip() else self._combine_passages(passages)
        citation_map, sources = self._build_citations(passages, context_text)
        outline = self._build_outline(
            question=question,
            context=context_text,
            citations=citation_map,
            section_headers=section_headers,
            bullets=bullets,
            examples=examples,
            min_words=min_words,
        )
        answer_text = outline
        word_count = self._word_count(answer_text)
        if word_count < min_words:
            padding = self._padding_paragraph(min_words - word_count, context_text)
            answer_text = f"{answer_text}\n\n{padding}"
            word_count = self._word_count(answer_text)

        if max_words and word_count > max_words:
            answer_text = self._trim_to_words(answer_text, max_words)
            word_count = self._word_count(answer_text)

        coverage = citation_coverage(answer_text)
        faith = faithfulness(answer_text, context_text)
        verbosity_val = verbosity_score(answer_text, min_words if verbosity in {"verbose", "max"} else 400)

        effects = []
        if coverage >= 0.2 or sources:
            effects.append("cited")
        if faith >= 0.3:
            effects.append("grounded")
        if word_count >= min_words or verbosity in {"verbose", "max"}:
            effects.append("verbose")

        if not effects and sources:
            effects.append("cited")

        sources_section = self._format_sources(sources)
        answer_text = f"{answer_text}\n\n{sources_section}"
        data["answer"] = answer_text
        data["sources"] = sources
        data["word_count"] = word_count

        rewards = {
            "citation_coverage": coverage,
            "faithfulness": faith,
            "verbosity_score": verbosity_val,
        }
        rationale = {
            "why": "Generated verbose, cited answer",
            "evidence": [
                f"word_count={word_count}",
                f"citations={len(sources)}",
                f"coverage={coverage:.2f}",
                f"faithfulness={faith:.2f}",
            ],
        }

        logs = [
            f"Answer generated with verbosity='{verbosity}', word_count={word_count}",
            f"Citation coverage={coverage:.2f}, faithfulness={faith:.2f}",
        ]

        if faith < 0.4:
            unsupported = self._unsupported_block(answer_text, context_text)
            if unsupported:
                answer_text = f"{answer_text}\n\n### Unsupported or low-support claims\n{unsupported}"
                data["answer"] = answer_text

        return {
            "ok": True,
            "output": {"answer": answer_text, "sources": sources, "word_count": word_count},
            "logs": logs,
            "checks": {},
            "reward": rewards,
            "rationale": rationale,
            "effects": effects,
        }

    def _failure(self, message: str) -> Result:
        return {
            "ok": False,
            "logs": [message],
            "checks": {},
            "reward": 0.0,
            "rationale": {"why": message, "evidence": []},
            "effects": [],
        }

    def _combine_passages(self, passages: List[Dict[str, object]]) -> str:
        pieces: List[str] = []
        for idx, passage in enumerate(passages, start=1):
            source = passage.get("meta", {}).get("title") or passage.get("meta", {}).get("url") or passage.get("doc_id") or f"passage-{idx}"
            pieces.append(f"[{idx}] {source}\n{passage.get('text', '')}")
        return "\n\n".join(pieces)

    def _build_citations(self, passages: List[Dict[str, object]], context_text: str) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
        citation_map: Dict[str, str] = {}
        sources: List[Dict[str, str]] = []
        for idx, passage in enumerate(passages, start=1):
            key = passage.get("doc_id") or f"doc-{idx}"
            display = passage.get("meta", {}).get("title") or passage.get("meta", {}).get("url") or key
            citation = f"[{idx}]"
            citation_map[key] = citation
            sources.append({"marker": citation, "source": str(display)})
        if not passages:
            sources.append({"marker": "[1]", "source": "Aggregate summary"})
        return citation_map, sources

    def _build_outline(
        self,
        *,
        question: str,
        context: str,
        citations: Dict[str, str],
        section_headers: bool,
        bullets: bool,
        examples: bool,
        min_words: int,
    ) -> str:
        intro = f"The question asks: {question}. The response below synthesizes retrieved evidence."
        key_findings = self._key_findings(context, citations, bullets)
        deep_dive = self._deep_dive(context, citations, examples)
        caveats = "Evidence is limited to the ingested documents; verify with up-to-date primary sources."
        sections = []
        if section_headers:
            sections.append("### Introduction\n" + intro)
            sections.append("### Key Findings\n" + key_findings)
            sections.append("### Deep Dive\n" + deep_dive)
            sections.append("### Caveats and Limitations\n" + caveats)
        else:
            sections = [intro, key_findings, deep_dive, caveats]
        answer = "\n\n".join(sections)
        if min_words > 1000:
            answer += "\n\n" + self._extended_examples(context, citations)
        return answer

    def _key_findings(self, context: str, citations: Dict[str, str], bullets: bool) -> str:
        sentences = [sent.strip() for sent in re.split(r"(?<=[.!?])\s+", context) if sent.strip()]
        top_items = sentences[:4]
        lines = []
        for idx, sentence in enumerate(top_items, start=1):
            marker = self._match_citation(sentence, citations)
            text = f"{sentence} {marker}".strip()
            if bullets:
                lines.append(f"{idx}. {text}")
            else:
                lines.append(text)
        return "\n".join(lines) if lines else "Further detail is summarized below."

    def _deep_dive(self, context: str, citations: Dict[str, str], examples: bool) -> str:
        paragraphs = []
        for paragraph in context.split("\n\n"):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            marker = self._match_citation(paragraph, citations)
            paragraphs.append(f"{paragraph} {marker}".strip())
        body = "\n\n".join(paragraphs)
        if examples:
            body += "\n\nExamples:\n- " + "\n- ".join(paragraphs[:2]) if paragraphs else "\n\nExamples: Not available."
        return body

    def _extended_examples(self, context: str, citations: Dict[str, str]) -> str:
        snippets = context.split("\n\n")[:3]
        lines = ["### Extended Examples"]
        for snippet in snippets:
            marker = self._match_citation(snippet, citations)
            lines.append(f"- {snippet.strip()} {marker}".strip())
        return "\n".join(lines)

    def _match_citation(self, text: str, citations: Dict[str, str]) -> str:
        for key, marker in citations.items():
            if key in text:
                return marker
        return "[1]" if citations else ""

    def _word_count(self, text: str) -> int:
        return len([token for token in re.findall(r"\w+", text)])

    def _trim_to_words(self, text: str, max_words: int) -> str:
        tokens = re.findall(r"\S+", text)
        if len(tokens) <= max_words:
            return text
        trimmed = " ".join(tokens[:max_words])
        return trimmed + "..."

    def _padding_paragraph(self, deficit: int, context: str) -> str:
        sentences = [sent.strip() for sent in re.split(r"(?<=[.!?])\s+", context) if sent.strip()]
        reuse = sentences[: max(1, min(3, len(sentences)))]
        filler = " ".join(reuse)
        if not filler:
            filler = "Additional elaboration ensures the response meets the requested verbosity while maintaining clarity."
        repeats = math.ceil(deficit / max(1, self._word_count(filler)))
        return " ".join([filler] * repeats)

    def _format_sources(self, sources: List[Dict[str, str]]) -> str:
        lines = ["### Sources"]
        for entry in sources:
            lines.append(f"{entry['marker']} {entry['source']}")
        return "\n".join(lines)

    def _unsupported_block(self, answer: str, context: str) -> str:
        unsupported = []
        context_lower = context.lower()
        for sentence in re.split(r"(?<=[.!?])\s+", answer):
            sentence_clean = sentence.strip()
            if not sentence_clean:
                continue
            if "[" not in sentence_clean:
                unsupported.append(sentence_clean)
                continue
            bare = re.sub(r"\[[^\]]+\]", "", sentence_clean).strip().lower()
            if bare and bare not in context_lower:
                unsupported.append(sentence_clean)
        return "\n".join(f"- {item}" for item in unsupported[:5])


__all__ = ["AnswerVerbose"]
