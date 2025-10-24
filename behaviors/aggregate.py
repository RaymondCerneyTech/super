from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from core.interfaces import Behavior, Context, Result


class AggregatePassages(Behavior):
    name = "aggregate"
    inputs = ["passages"]
    outputs = ["aggregated_text"]

    def run(self, ctx: Context) -> Result:
        data = ctx.setdefault("data", {})
        passages = data.get("passages") or ctx.get("passages") or []
        strategy = (data.get("aggregate_strategy") or ctx.get("aggregate_strategy") or "concatenate").lower()
        if not isinstance(passages, list) or not passages:
            return {
                "ok": False,
                "logs": ["No passages available to aggregate."],
                "checks": {},
                "reward": 0.0,
                "rationale": {"why": "missing passages", "evidence": []},
                "effects": [],
            }

        aggregated_text = ""
        evidence: List[str] = []
        if strategy == "topic_blocks":
            aggregated_text, block_count = self._topic_blocks(passages)
            evidence.append(f"blocks={block_count}")
        else:
            aggregated_text, count = self._concatenate(passages)
            evidence.append(f"passages={count}")

        data["aggregated_text"] = aggregated_text

        rationale = {
            "why": f"Aggregated passages using {strategy}",
            "evidence": evidence,
        }
        reward = 1.0 if aggregated_text else 0.2

        return {
            "ok": True,
            "output": {"aggregated_text": aggregated_text},
            "logs": [f"Aggregated passages with strategy='{strategy}'"],
            "checks": {},
            "reward": reward,
            "rationale": rationale,
            "effects": ["has_aggregate"],
        }

    def _concatenate(self, passages: List[Dict[str, object]]) -> tuple[str, int]:
        pieces: List[str] = []
        for idx, passage in enumerate(passages, start=1):
            header = passage.get("meta", {}).get("title") or passage.get("meta", {}).get("url") or passage.get("doc_id") or f"passage-{idx}"
            pieces.append(f"## Source {idx}: {header}")
            pieces.append(str(passage.get("text", "")))
        return "\n\n".join(pieces), len(passages)

    def _topic_blocks(self, passages: List[Dict[str, object]]) -> tuple[str, int]:
        clusters: Dict[str, List[str]] = defaultdict(list)
        for passage in passages:
            text = str(passage.get("text", ""))
            key = self._topic_key(text)
            clusters[key].append(text)
        pieces: List[str] = []
        for idx, (topic, snippets) in enumerate(clusters.items(), start=1):
            pieces.append(f"## Topic {idx}: {topic}")
            pieces.append("\n\n".join(snippets))
        return "\n\n".join(pieces), len(clusters)

    def _topic_key(self, text: str) -> str:
        keywords = ["security", "performance", "business", "finance", "product", "research", "people"]
        text_lower = text.lower()
        for keyword in keywords:
            if keyword in text_lower:
                return keyword
        return "general"


__all__ = ["AggregatePassages"]
