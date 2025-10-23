# core/checks.py
from typing import Dict, Any
import math

TONE_KEYWORDS = {
    "professional": ["please", "regards", "appreciate", "sincerely"],
    "casual": ["hey", "thanks", "cheers", "awesome"],
}

def length_leq(output: str, N: int) -> float:
    return 1.0 if len(output.split()) <= N else 0.0

def similarity_cosine(a: str, b: str) -> float:
    # placeholder: token-set Jaccard as a stand-in for cosine
    A, B = set(a.lower().split()), set(b.lower().split())
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)  # 0..1

def tone_keyword_match(output: str, tone: str) -> float:
    keywords = TONE_KEYWORDS.get((tone or "").lower())
    if not keywords:
        return 0.0
    text = output.lower()
    hits = sum(1 for kw in keywords if kw in text)
    return hits / len(keywords)

CHECKS = {
    "length_leq": length_leq,
    "similarity_cosine": similarity_cosine,
    "tone_keyword_match": tone_keyword_match,
}
