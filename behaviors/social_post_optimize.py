from __future__ import annotations

import re
from typing import Dict, List

from core.interfaces import Behavior, Context, Result

PLATFORM_HINTS: Dict[str, Dict[str, List[str]]] = {
    "twitter": {
        "tone": ["concise", "engaging"],
        "call_to_action": ["Retweet", "Follow"],
    },
    "linkedin": {
        "tone": ["professional", "insightful"],
        "call_to_action": ["Learn more", "Connect"],
    },
    "instagram": {
        "tone": ["warm", "visual"],
        "call_to_action": ["Share", "Save"],
    },
}

HASHTAG_CANDIDATES = {
    "business": ["#BusinessTips", "#Leadership", "#Strategy"],
    "product": ["#ProductLaunch", "#Innovation", "#Tech"],
    "marketing": ["#Marketing", "#Branding", "#Growth"],
}


class SocialPostOptimize(Behavior):
    name = "social_post_optimize"
    inputs = ["text"]
    outputs = ["optimized_text", "hashtags"]

    def run(self, ctx: Context) -> Result:
        data = ctx.setdefault("data", {})
        text = (data.get("text") or ctx.get("text") or "").strip()
        platform = (data.get("platform") or ctx.get("platform") or "twitter").lower()
        topic = (data.get("topic") or ctx.get("topic") or "business").lower()
        max_length = int(data.get("max_length") or (280 if platform == "twitter" else 400))

        optimized = self._refine_text(text, platform, max_length)
        hashtags = self._select_hashtags(topic, platform)

        data["optimized_text"] = optimized
        data["hashtags"] = hashtags

        logs = [f"Optimized post for platform='{platform}' with topic='{topic}'."]
        return {
            "ok": True,
            "output": {"optimized_text": optimized, "hashtags": hashtags},
            "logs": logs,
            "checks": {},
            "reward": 0.0,
        }

    def _refine_text(self, text: str, platform: str, max_length: int) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        hints = PLATFORM_HINTS.get(platform, {})
        tone_words = hints.get("tone", [])
        call_to_actions = hints.get("call_to_action", [])

        if tone_words and tone_words[0].lower() not in text.lower():
            text = f"{tone_words[0].capitalize()}: {text}"
        if call_to_actions:
            text = f"{text} {call_to_actions[0]}!"

        if len(text) > max_length:
            text = text[:max_length - 3].rstrip() + "..."
        return text

    def _select_hashtags(self, topic: str, platform: str) -> List[str]:
        tags = HASHTAG_CANDIDATES.get(topic, [])[:3]
        if platform == "twitter" and len(tags) > 2:
            tags = tags[:2]
        return tags


__all__ = ["SocialPostOptimize"]
