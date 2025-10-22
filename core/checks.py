# core/checks.py
from typing import Dict, Any
import math

def length_leq(output: str, N: int) -> float:
    return 1.0 if len(output.split()) <= N else 0.0

def similarity_cosine(a: str, b: str) -> float:
    # placeholder: token-set Jaccard as a stand-in for cosine
    A, B = set(a.lower().split()), set(b.lower().split())
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)  # 0..1

CHECKS = {
    "length_leq": length_leq,
    "similarity_cosine": similarity_cosine,
}
