from __future__ import annotations

from typing import Dict, List, Tuple

from core.interfaces import Behavior, Context, Result
from core.rewards import ensure_reward_dict

POSITIVE_WORDS = {"good", "great", "excellent", "love", "amazing", "happy", "joy", "delightful", "fantastic", "awesome"}
NEGATIVE_WORDS = {"bad", "terrible", "awful", "hate", "sad", "angry", "horrible", "disappointing", "furious"}

EMOTION_LEXICON: Dict[str, set[str]] = {
    "joy": {"joy", "happy", "delighted", "excited", "glad", "cheerful", "smile"},
    "anger": {"angry", "furious", "irritated", "annoyed", "rage", "mad"},
    "sadness": {"sad", "down", "depressed", "unhappy", "miserable"},
    "fear": {"afraid", "scared", "fearful", "terrified", "worried", "nervous"},
    "surprise": {"surprised", "shocked", "astonished", "amazed"},
}

SARCASM_CUES = {"yeah right", "sure", "as if", "totally", "could not be happier"}
NEGATORS = {"not", "never", "no", "hardly", "barely"}
INTENSIFIERS = {"very", "extremely", "incredibly", "totally", "really", "so"}


class SentimentAnalysis(Behavior):
    name = "sentiment_analysis"
    inputs = ["text"]
    outputs = ["sentiment", "sentiment_score", "sentiment_category", "sentiment_intensity", "analysis_text"]

    def run(self, ctx: Context) -> Result:
        data = ctx.setdefault("data", {})
        text = (data.get("text") or ctx.get("text") or "").strip()
        label, category, intensity, score, explanation = self._analyze(text)

        data["sentiment"] = label
        data["sentiment_category"] = category
        data["sentiment_intensity"] = intensity
        data["sentiment_score"] = score
        data["analysis_text"] = explanation

        return {
            "ok": True,
            "output": {
                "sentiment": label,
                "sentiment_category": category,
                "sentiment_intensity": intensity,
                "sentiment_score": score,
                "analysis_text": explanation,
            },
            "logs": [f"Sentiment detected: {label}/{category} (intensity={intensity:.2f}, score={score:.2f})."],
            "checks": {},
            "reward": 0.0,
        }

    def _analyze(self, text: str) -> Tuple[str, str, float, float, str]:
        if not text:
            return "neutral", "neutral", 0.0, 0.5, "No content provided; defaulting to neutral sentiment."

        tokens = [token.strip(".,!?;:\"'").lower() for token in text.split() if token]
        token_counts = {token: tokens.count(token) for token in set(tokens)}

        positive_hits = sum(token_counts.get(word, 0) for word in POSITIVE_WORDS)
        negative_hits = sum(token_counts.get(word, 0) for word in NEGATIVE_WORDS)
        negators_present = any(token in NEGATORS for token in tokens)
        intensifiers_present = sum(token in INTENSIFIERS for token in tokens)

        emotion_scores: Dict[str, int] = {}
        for emotion, lexicon in EMOTION_LEXICON.items():
            emotion_scores[emotion] = sum(token_counts.get(word, 0) for word in lexicon)

        sarcasm_detected = self._detect_sarcasm(text.lower(), positive_hits, negative_hits, negators_present)

        polarity = positive_hits - negative_hits
        total_hits = positive_hits + negative_hits or 1
        base_score = polarity / total_hits
        intensity = min(1.0, max(0.0, abs(base_score) + intensifiers_present * 0.1))

        if sarcasm_detected:
            label = "negative"
            category = "sarcasm"
            score = 0.5
        else:
            if base_score > 0.1:
                label = "positive"
            elif base_score < -0.1:
                label = "negative"
            else:
                label = "neutral"

            if emotion_scores:
                category = max(emotion_scores.items(), key=lambda item: item[1] or -1)[0]
            else:
                category = label

            score = min(1.0, 0.5 + abs(base_score) / 2 + intensifiers_present * 0.05)

        explanation = self._build_explanation(label, category, intensity, score, tokens)
        return label, category, intensity, score, explanation

    def _detect_sarcasm(self, text: str, positive_hits: int, negative_hits: int, negators_present: bool) -> bool:
        if any(phrase in text for phrase in SARCASM_CUES):
            return True
        if positive_hits > 0 and negative_hits > 0:
            return True
        if positive_hits > 0 and negators_present:
            return True
        return False

    def _build_explanation(
        self,
        label: str,
        category: str,
        intensity: float,
        score: float,
        tokens: List[str],
    ) -> str:
        summary = f"The sentiment is {label} with category {category}. "
        summary += f"Intensity {intensity:.2f}, confidence {score:.2f}. "
        summary += f"Sample highlights: {', '.join(tokens[:5])}."
        return summary


__all__ = ["SentimentAnalysis"]
