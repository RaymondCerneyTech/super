from __future__ import annotations

from typing import Dict

from core.interfaces import Behavior, Context, Result

POSITIVE_WORDS = {"good", "great", "excellent", "love", "amazing", "happy"}
NEGATIVE_WORDS = {"bad", "terrible", "awful", "hate", "sad", "angry"}


class SentimentAnalysis(Behavior):
    name = "sentiment_analysis"
    inputs = ["text"]
    outputs = ["sentiment", "sentiment_score"]

    def run(self, ctx: Context) -> Result:
        data = ctx.setdefault("data", {})
        text = (data.get("text") or ctx.get("text") or "").strip()
        label, score = self._analyze(text)
        data["sentiment"] = label
        data["sentiment_score"] = score
        return {
            "ok": True,
            "output": {"sentiment": label, "sentiment_score": score},
            "logs": [f"Classified sentiment as {label} (score={score:.2f})."],
            "checks": {},
            "reward": 0.0,
        }

    def _analyze(self, text: str) -> tuple[str, float]:
        if not text:
            return "neutral", 0.5
        tokens = [token.strip(".,!?;:").lower() for token in text.split() if token]
        pos_hits = sum(token in POSITIVE_WORDS for token in tokens)
        neg_hits = sum(token in NEGATIVE_WORDS for token in tokens)
        total = pos_hits + neg_hits
        if total == 0:
            return "neutral", 0.5
        polarity = (pos_hits - neg_hits) / total
        if polarity > 0.1:
            return "positive", min(1.0, 0.5 + polarity / 2)
        if polarity < -0.1:
            return "negative", min(1.0, 0.5 + abs(polarity) / 2)
        return "neutral", 0.5


__all__ = ["SentimentAnalysis"]
