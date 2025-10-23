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
        mode = (data.get("grammar_mode") or ctx.get("grammar_mode") or "standard").lower()
        aggressive = mode in {"aggressive", "deep"}

        corrected = self._correct_text(text, aggressive=aggressive)
        data["corrected_text"] = corrected

        logs = []
        if corrected != text:
            logs.append("Grammar issues corrected.")
        else:
            logs.append("No grammar changes were necessary.")
        if aggressive:
            logs.append("Aggressive grammar refinement enabled.")

        return {
            "ok": True,
            "output": {"corrected_text": corrected},
            "logs": logs,
            "checks": {},
            "reward": 0.0,
        }

    def _correct_text(self, text: str, aggressive: bool = False) -> str:
        if not text:
            return ""

        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
        corrected_sentences = []
        for sentence in sentences:
            sentence = self._basic_normalization(sentence)
            if aggressive:
                sentence = self._style_polish(sentence)
            corrected_sentences.append(sentence)

        corrected = " ".join(corrected_sentences)
        if aggressive:
            corrected = self._ensure_terminal_punctuation(corrected)
        return corrected

    def _basic_normalization(self, sentence: str) -> str:
        sentence = re.sub(r"\bi\b", "I", sentence)
        sentence = re.sub(r"\bi'm\b", "I'm", sentence, flags=re.IGNORECASE)
        sentence = re.sub(r"\s+", " ", sentence)
        if sentence and sentence[0].islower():
            sentence = sentence[0].upper() + sentence[1:]
        return sentence

    def _style_polish(self, sentence: str) -> str:
        sentence = re.sub(r"\b(can not)\b", "cannot", sentence, flags=re.IGNORECASE)
        sentence = re.sub(r"\b(won't)\b", "will not", sentence, flags=re.IGNORECASE)
        sentence = re.sub(r"\b(don't)\b", "do not", sentence, flags=re.IGNORECASE)
        sentence = re.sub(r"\b(gonna)\b", "going to", sentence, flags=re.IGNORECASE)
        sentence = re.sub(r"\b(wanna)\b", "want to", sentence, flags=re.IGNORECASE)
        sentence = re.sub(r"\b(kinda)\b", "kind of", sentence, flags=re.IGNORECASE)
        sentence = re.sub(r"([!?]){2,}", r"\1", sentence)
        sentence = re.sub(r"\s*,\s*,", ",", sentence)
        sentence = re.sub(r"\s*([,.!?;:])", r"\1", sentence)
        sentence = re.sub(r"([,.!?;:])([^\s])", r"\1 \2", sentence)
        return sentence.strip()

    def _ensure_terminal_punctuation(self, text: str) -> str:
        if not text:
            return ""
        if text[-1] not in ".!?":
            return text + "."
        return text


__all__ = ["GrammarCorrection"]
