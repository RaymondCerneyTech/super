from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from judges.meta_judge import aggregate


META_REVIEW_PATH = Path(".ai") / "meta" / "reviewer.json"


@dataclass
class Hypothesis:
    id: str
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "meta": dict(self.meta),
            "notes": list(self.notes),
        }


class GenerationAgent:
    def __init__(self) -> None:
        self.templates = [
            "Frame the {focus} problem as a {angle} systems experiment that instruments flaky signals end-to-end.",
            "Operationalize {focus} via a {angle} intervention that runs weekly guardrails and publishes visible health metrics.",
            "Prototype a {angle} assistant that pairs subject-matter experts with automated checks to derisk {focus}.",
            "Use cohort comparisons to stress {focus} with synthetic failures, then publish a {angle} playbook for mitigation.",
        ]
        self.angles = [
            "mechanistic",
            "data-informed",
            "workflow",
            "simulation",
            "control chart",
        ]

    def generate(
        self,
        prompt: str,
        *,
        count: int = 4,
        seeds: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, Any]]:
        hypotheses: List[Dict[str, Any]] = []
        cleaned_prompt = " ".join(prompt.split())
        seed_list = [s for s in (seeds or []) if isinstance(s, str) and s.strip()]
        total = max(count, len(seed_list))
        for idx in range(total):
            hyp_id = f"H{idx + 1}"
            if idx < len(seed_list):
                text = seed_list[idx].strip()
                angle = "seeded"
            else:
                template = self.templates[idx % len(self.templates)]
                angle = self.angles[idx % len(self.angles)]
                text = template.format(focus=cleaned_prompt.lower(), angle=angle)
            hypotheses.append(
                {
                    "id": hyp_id,
                    "text": text,
                    "meta": {
                        "theme": angle,
                        "focus": cleaned_prompt,
                        "novelty": round(0.55 + 0.05 * (idx % 5), 2),
                        "evidence_need": 0.6 if "instrument" in text.lower() else 0.4,
                    },
                    "notes": [],
                }
            )
        return hypotheses


class ReflectionAgent:
    DIMENSIONS = ("clarity", "evidence", "risk")

    def reflect(self, hypotheses: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        reflections: List[Dict[str, Any]] = []
        for entry in hypotheses:
            text = str(entry.get("text", "")).strip()
            length = len(text)
            clarity = min(1.0, length / 280.0)
            evidence = 0.7 if "measure" in text.lower() or "instrument" in text.lower() else 0.45
            risk = 0.3 if "simulate" in text.lower() or "synthetic" in text.lower() else 0.6
            dimensions = {
                "clarity": round(clarity, 2),
                "evidence": round(evidence, 2),
                "risk": round(1.0 - risk, 2),
            }
            critiques = []
            if clarity < 0.4:
                critiques.append("Clarify the intervention in one crisp sentence.")
            if evidence < 0.5:
                critiques.append("Add instrumentation or observational checkpoints.")
            if risk > 0.5:
                critiques.append("Call out a fail-fast gate to limit downside.")
            reflections.append(
                {
                    "id": entry.get("id"),
                    "critiques": critiques,
                    "dimensions": dimensions,
                }
            )
        return reflections


class RankingAgent:
    def __init__(self, base_rating: float = 1200.0, k_factor: float = 24.0) -> None:
        self.base_rating = base_rating
        self.k_factor = k_factor

    def rank(
        self,
        hypotheses: List[Dict[str, Any]],
        *,
        question: str = "",
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not hypotheses:
            return [], {"elo": {}, "matches": []}

        ratings = {entry["id"]: self.base_rating for entry in hypotheses if entry.get("id")}
        matches: List[Dict[str, Any]] = []
        total = len(hypotheses)
        for i in range(total):
            for j in range(i + 1, total):
                a = hypotheses[i]
                b = hypotheses[j]
                pair = [a, b]
                winner_idx, judge_scores = aggregate([self._to_candidate(a, question), self._to_candidate(b, question)])
                winner = pair[winner_idx]
                loser = pair[1 - winner_idx]
                w_id = winner["id"]
                l_id = loser["id"]
                if w_id not in ratings or l_id not in ratings:
                    continue
                expected_w = self._expected(ratings[w_id], ratings[l_id])
                expected_l = self._expected(ratings[l_id], ratings[w_id])
                ratings[w_id] = ratings[w_id] + self.k_factor * (1 - expected_w)
                ratings[l_id] = ratings[l_id] + self.k_factor * (0 - expected_l)
                matches.append(
                    {
                        "winner": w_id,
                        "loser": l_id,
                        "judge": judge_scores,
                    }
                )

        enriched = []
        for entry in hypotheses:
            hyp_copy = dict(entry)
            hyp_copy["rating"] = round(ratings.get(entry["id"], self.base_rating), 2)
            enriched.append(hyp_copy)
        ordered = sorted(enriched, key=lambda x: x.get("rating", 0.0), reverse=True)
        champion_id = ordered[0]["id"]
        ranking_meta = {
            "elo": ratings,
            "matches": matches,
            "champion": champion_id,
            "champion_rating": ratings.get(champion_id, self.base_rating),
        }
        return ordered, ranking_meta

    def _to_candidate(self, hypothesis: Dict[str, Any], question: str) -> Dict[str, Any]:
        score = self._estimate_reward(hypothesis, question)
        return {
            "effects": ["compliant"],
            "rewards": {"overall": score},
            "output": {"final": hypothesis.get("text", "")},
        }

    def _estimate_reward(self, hypothesis: Dict[str, Any], question: str) -> float:
        text = str(hypothesis.get("text", "")).lower()
        reward = 0.5
        if "measure" in text or "instrument" in text:
            reward += 0.15
        if "simulate" in text or "synthetic" in text:
            reward += 0.1
        if question and any(token in text for token in question.lower().split()):
            reward += 0.1
        reward += min(0.15, len(text) / 800.0)
        return max(0.0, min(1.0, reward))

    def _expected(self, rating_a: float, rating_b: float) -> float:
        exponent = (rating_b - rating_a) / 400.0
        return 1.0 / (1.0 + math.pow(10.0, exponent))


class EvolutionAgent:
    def evolve(
        self,
        hypotheses: List[Dict[str, Any]],
        *,
        max_variants: int = 1,
    ) -> List[Dict[str, Any]]:
        if len(hypotheses) < 2 or max_variants <= 0:
            return []
        variants: List[Dict[str, Any]] = []
        top = hypotheses[: max(2, max_variants + 1)]
        for idx in range(min(max_variants, len(top) - 1)):
            primary = top[idx]
            secondary = top[idx + 1]
            blended = (
                f"Hybridize {primary['id']} ({primary.get('meta', {}).get('theme', 'core')}) "
                f"with {secondary['id']} to run dual-track instrumentation plus resilience rehearsal."
            )
            variants.append(
                {
                    "id": f"{primary['id']}x{secondary['id']}",
                    "text": blended,
                    "meta": {
                        "origin": "evolved",
                        "parents": [primary["id"], secondary["id"]],
                        "novelty": 0.8,
                    },
                    "notes": ["Merged top contenders"],
                }
            )
        return variants


class MetaReviewAgent:
    def review(
        self,
        *,
        prompt: str,
        champion: Optional[Dict[str, Any]],
        reflections: List[Dict[str, Any]],
        ranking_meta: Dict[str, Any],
        verifier_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        review = {
            "prompt": prompt,
            "champion": champion.get("id") if champion else None,
            "verifier_score": verifier_score,
            "champion_text": champion.get("text") if champion else "",
            "matches": ranking_meta.get("matches", []),
        }
        weights = self._load_weights()
        champion_dims = self._dimensions_for(champion, reflections)
        for dim, value in champion_dims.items():
            current = weights.get(dim, 0.5)
            weights[dim] = round(0.7 * current + 0.3 * value, 4)
        self._save_weights(weights, review)
        review["weights"] = weights
        review["dimensions"] = champion_dims
        return review

    def _dimensions_for(
        self,
        champion: Optional[Dict[str, Any]],
        reflections: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        if not champion:
            return {}
        lookup = {entry.get("id"): entry for entry in reflections}
        entry = lookup.get(champion.get("id"))
        if not entry:
            return {}
        dims = entry.get("dimensions") or {}
        return {str(k): float(v) for k, v in dims.items()}

    def _load_weights(self) -> Dict[str, float]:
        if META_REVIEW_PATH.exists():
            try:
                payload = json.loads(META_REVIEW_PATH.read_text(encoding="utf-8"))
                return dict(payload.get("weights", {}))
            except (OSError, json.JSONDecodeError):
                return {}
        return {}

    def _save_weights(self, weights: Dict[str, float], review: Dict[str, Any]) -> None:
        META_REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "weights": weights,
            "last_review": review,
        }
        try:
            META_REVIEW_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            pass


__all__ = [
    "EvolutionAgent",
    "GenerationAgent",
    "Hypothesis",
    "MetaReviewAgent",
    "RankingAgent",
    "ReflectionAgent",
]
