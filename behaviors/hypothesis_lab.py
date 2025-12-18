from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agents import EvolutionAgent, GenerationAgent, MetaReviewAgent, RankingAgent, ReflectionAgent
from core.interfaces import Behavior, Context, Result
from core.rewards import json_schema_ok, math_exact, unit_tests_pass


class HypothesisLab(Behavior):
    name = "hypothesis_lab"
    inputs = ["question"]
    outputs = ["hypotheses"]

    def __init__(self) -> None:
        self.generator = GenerationAgent()
        self.reflector = ReflectionAgent()
        self.ranker = RankingAgent()
        self.evolver = EvolutionAgent()
        self.reviewer = MetaReviewAgent()

    def run(self, ctx: Context) -> Result:
        data = ctx.setdefault("data", {})
        prompt = str(
            data.get("question")
            or ctx.get("question")
            or ctx.get("text")
            or data.get("text")
            or ctx.get("goal")
            or ""
        ).strip()
        if not prompt:
            return self._failure("Hypothesis lab requires a question or prompt.")

        count = max(2, int(data.get("hypothesis_count") or 4))
        seeds = data.get("seed_hypotheses")
        hypotheses = self.generator.generate(prompt, count=count, seeds=seeds)

        reflections = self.reflector.reflect(hypotheses)
        self._merge_reflections(hypotheses, reflections)

        ranked, ranking_meta = self.ranker.rank(hypotheses, question=prompt)

        evolutions: List[Dict[str, Any]] = []
        if data.get("enable_evolution", True):
            evolutions = self.evolver.evolve(ranked, max_variants=int(data.get("max_evolutions", 1)))
            if evolutions:
                combined = ranked + evolutions
                ranked, ranking_meta = self.ranker.rank(combined, question=prompt)

        champion = ranked[0] if ranked else None
        verifier_cfg = data.get("hypothesis_verifier")
        verifier_score = self._run_verifier(champion, verifier_cfg)

        review = self.reviewer.review(
            prompt=prompt,
            champion=champion,
            reflections=reflections,
            ranking_meta=ranking_meta,
            verifier_score=verifier_score,
        )

        data["hypotheses"] = ranked
        data["hypothesis_reflections"] = reflections
        data["hypothesis_ranking"] = ranking_meta
        data["hypothesis_review"] = review
        data["evolved_hypotheses"] = evolutions

        rationale = {
            "why": "Generated, critiqued, and ranked hypotheses via mini co-scientist loop.",
            "evidence": [
                f"hypotheses={len(ranked)}",
                f"champion={ranking_meta.get('champion')}",
                f"verifier={verifier_score if verifier_score is not None else 'n/a'}",
            ],
        }
        reward = min(1.0, ranking_meta.get("champion_rating", 1200.0) / 1600.0)
        rewards = {"overall": reward}
        if verifier_score is not None:
            rewards["verifier"] = verifier_score

        logs = [
            f"[hypothesis_lab] generated {len(hypotheses)} hypotheses",
            f"[hypothesis_lab] champion={ranking_meta.get('champion')} rating={ranking_meta.get('champion_rating', 0):.1f}",
        ]
        if evolutions:
            logs.append(f"[hypothesis_lab] evolved {len(evolutions)} variants")

        return {
            "ok": True,
            "output": {"hypotheses": ranked},
            "logs": logs,
            "effects": ["has_hypotheses", "has_meta_review"],
            "rewards": rewards,
            "rationale": rationale,
        }

    def _merge_reflections(self, hypotheses: List[Dict[str, Any]], reflections: List[Dict[str, Any]]) -> None:
        lookup = {entry.get("id"): entry for entry in hypotheses}
        for reflection in reflections:
            target = lookup.get(reflection.get("id"))
            if not target:
                continue
            notes = target.setdefault("notes", [])
            notes.extend(reflection.get("critiques", []))
            if reflection.get("dimensions"):
                dims = target.setdefault("dimensions", {})
                dims.update(reflection["dimensions"])

    def _run_verifier(self, champion: Optional[Dict[str, Any]], cfg: Any) -> Optional[float]:
        if not cfg:
            return None
        verifier_type = str(cfg.get("type") or "").lower()
        if not verifier_type:
            return None
        if verifier_type == "math_exact":
            target = cfg.get("target")
            value = cfg.get("value")
            if value is None and champion:
                metric_key = cfg.get("metric")
                if metric_key and isinstance(champion.get("meta"), dict):
                    value = champion["meta"].get(metric_key)
                else:
                    value = len(champion.get("text", ""))
            return math_exact(value, target)
        if verifier_type == "json_schema_ok":
            schema = cfg.get("schema")
            sample = cfg.get("value")
            if sample is None and champion:
                sample = {
                    "hypothesis": champion.get("text"),
                    "id": champion.get("id"),
                }
            return json_schema_ok(sample, schema)
        if verifier_type == "unit_tests_pass":
            report = cfg.get("report")
            return unit_tests_pass(report)
        return None

    def _failure(self, message: str) -> Result:
        return {
            "ok": False,
            "logs": [message],
            "rewards": {"overall": 0.0},
            "effects": [],
            "rationale": {"why": message, "evidence": []},
        }


__all__ = ["HypothesisLab"]
