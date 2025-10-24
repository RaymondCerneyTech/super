from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Tuple

from core.interfaces import Behavior, Context, Result


class Summarize(Behavior):
    name = "summarize"
    inputs = ["text"]
    outputs = ["summary"]

    def run(self, ctx: Context) -> Result:
        data = ctx.setdefault("data", {})
        text = (data.get("text") or ctx.get("text") or "").strip()
        max_words = int(data.get("max_words") or 120)
        max_sentences = int(data.get("max_sentences") or 5)
        strategy = (data.get("summary_strategy") or ctx.get("summary_strategy") or "extractive").lower()

        summary, evidence = self._apply_strategy(text, strategy, max_sentences, max_words)
        data.setdefault("summary", summary)

        rationale = {
            "why": f"Selected summary using {strategy} strategy",
            "evidence": evidence,
        }

        logs = [f"Summarization strategy: {strategy}"]
        if len(summary.split()) > max_words:
            logs.append("Summary exceeded max_words after processing; trimming applied.")
            summary = self._trim_summary(summary, max_words)
            data["summary"] = summary

        effects = ["have_summary"]
        if summary:
            summary_tokens = summary.split()
            original_tokens = text.split()
            if summary_tokens and len(summary_tokens) <= max_words:
                effects.append("concise")
            if self._is_exact_match(summary, text, strategy):
                effects.append("exact")

        return {
            "ok": True,
            "output": {"summary": summary},
            "logs": logs,
            "checks": {},
            "reward": 0.0,
            "rationale": rationale,
            "effects": effects,
        }

    def _apply_strategy(
        self,
        text: str,
        strategy: str,
        max_sentences: int,
        max_words: int,
    ) -> Tuple[str, List[str]]:
        if not text:
            return "", []
        if strategy == "extractive":
            return self._extractive_summary(text, max_sentences, max_words)
        if strategy == "abstractive":
            return self._abstractive_summary(text, max_sentences, max_words)
        trimmed = self._trim_summary(text, max_words)
        return trimmed, [trimmed]

    def _trim_summary(self, text: str, max_words: int) -> str:
        if max_words <= 0:
            return ""
        words = text.split()
        return " ".join(words[:max_words])

    def _extractive_summary(self, text: str, max_sentences: int, max_words: int) -> Tuple[str, List[str]]:
        sentences = self._split_sentences(text)
        if not sentences:
            return "", []
        if len(sentences) <= max_sentences:
            trimmed = self._trim_summary(text, max_words)
            return trimmed, sentences

        word_scores = self._word_importance(sentences)
        scored_sentences = []
        for idx, sentence in enumerate(sentences):
            tokens = self._tokenize(sentence)
            if not tokens:
                continue
            score = sum(word_scores.get(token, 0.0) for token in tokens)
            normalization = math.log(len(tokens) + 1)
            scored_sentences.append((score / normalization if normalization else score, idx, sentence))

        scored_sentences.sort(reverse=True)
        selected = sorted(scored_sentences[:max_sentences], key=lambda item: item[1])
        chosen_sentences = [sentence for _, _, sentence in selected]
        summary = " ".join(chosen_sentences)
        summary = self._trim_summary(summary, max_words)
        return summary, chosen_sentences

    def _abstractive_summary(self, text: str, max_sentences: int, max_words: int) -> Tuple[str, List[str]]:
        sentences = self._split_sentences(text)
        if not sentences:
            return "", []
        first = sentences[0]
        last = sentences[-1] if len(sentences) > 1 else ""
        mid_summary, mid_evidence = self._extractive_summary(text, max_sentences=max(1, max_sentences - 1), max_words=max_words)
        pieces = [phrase for phrase in [first, mid_summary, last] if phrase]
        summary = " ".join(pieces)
        summary = self._trim_summary(summary, max_words)
        evidence = [first] + mid_evidence + ([last] if last else [])
        return summary, evidence

    def _split_sentences(self, text: str) -> List[str]:
        raw_sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = []
        for sentence in raw_sentences:
            cleaned = sentence.strip()
            if cleaned:
                sentences.append(cleaned)
        return sentences

    def _tokenize(self, sentence: str) -> List[str]:
        return re.findall(r"\w+", sentence.lower())

    def _word_importance(self, sentences: Iterable[str]) -> Dict[str, float]:
        tokens = []
        for sentence in sentences:
            tokens.extend(self._tokenize(sentence))
        frequency = Counter(tokens)
        if not frequency:
            return {}
        max_freq = max(frequency.values())
        return {word: freq / max_freq for word, freq in frequency.items()}

    def _is_exact_match(self, summary: str, original: str, strategy: str) -> bool:
        if not summary or not original:
            return False
        normalized_summary = re.sub(r"\s+", " ", summary.strip().lower())
        normalized_original = re.sub(r"\s+", " ", original.strip().lower())
        if strategy in {"trim", "extractive"} and normalized_summary in normalized_original:
            return True
        # fall back to word coverage heuristic
        summary_tokens = set(self._tokenize(summary))
        original_tokens = set(self._tokenize(original))
        if not summary_tokens:
            return False
        overlap = len(summary_tokens & original_tokens) / len(summary_tokens)
        return overlap >= 0.9 and strategy != "abstractive"


__all__ = ["Summarize"]
