from __future__ import annotations

import re
from typing import Any, Dict

STYLE_PATTERN = re.compile(r"\b(professional|casual|rewrite|tone)\b", re.IGNORECASE)
COMPRESS_PATTERN = re.compile(
    r"\b(condense|shorten|summarize|tl;dr|fewer words|compress)\b", re.IGNORECASE
)


def extract_feature_key(ctx: Dict[str, Any]) -> str:
    text = (
        ctx.get("text")
        or ctx.get("data", {}).get("text")
        or ""
    )
    if not isinstance(text, str):
        text = str(text)

    word_count = len(text.split())
    has_style = bool(STYLE_PATTERN.search(text))
    has_compress_kw = bool(COMPRESS_PATTERN.search(text))

    max_words = ctx.get("data", {}).get("max_words") if isinstance(ctx.get("data"), dict) else None
    long_by_size = word_count >= 80
    long_by_ratio = (
        isinstance(max_words, int) and max_words > 0 and word_count > 1.5 * max_words
    )

    if has_style:
        return "style_hint"
    if has_compress_kw or long_by_ratio:
        return "compress_hint"
    if long_by_size:
        return "long_text"
    return "default"


__all__ = ["extract_feature_key"]
