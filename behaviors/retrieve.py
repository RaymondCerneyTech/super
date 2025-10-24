from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from core.interfaces import Behavior, Context, Result
from core.index import get_index


def _parse_tags(raw: object) -> List[str]:
    if not raw:
        return []
    if isinstance(raw, str):
        items: Iterable[str] = raw.split(",")
    elif isinstance(raw, Iterable):
        items = [str(x) for x in raw]  # type: ignore[arg-type]
    else:
        items = [str(raw)]
    return [item.strip().lower() for item in items if str(item).strip()]


class Retrieve(Behavior):
    name = "retrieve"
    inputs = ["question"]
    outputs = ["passages"]

    def run(self, ctx: Context) -> Result:
        data = ctx.setdefault("data", {})
        question = (
            data.get("question")
            or ctx.get("question")
            or data.get("query")
            or ctx.get("query")
            or data.get("text")
            or ctx.get("text")
            or ""
        )
        question = str(question).strip()
        if not question:
            return {
                "ok": False,
                "logs": ["No question or query provided for retrieval."],
                "checks": {},
                "reward": 0.0,
                "rationale": {"why": "missing question", "evidence": []},
                "effects": [],
            }

        backend = (data.get("index_backend") or ctx.get("index_backend") or "tfidf").lower()
        index = get_index(backend=backend)
        k_passages = int(data.get("k_passages") or ctx.get("k_passages") or 12)
        max_chars = int(data.get("max_chars") or ctx.get("max_chars") or 12000)
        tags = _parse_tags(data.get("tags") or ctx.get("tags"))
        fresh_days_raw = data.get("fresh_days") or ctx.get("fresh_days")
        recency_days = int(fresh_days_raw) if fresh_days_raw else None

        results = index.search(question, k=k_passages, tags=tags, recency_days=recency_days)
        passages: List[Dict[str, object]] = []
        total_chars = 0
        recency_values: List[float] = []
        for entry in results:
            text = str(entry.get("text", ""))
            remaining = max_chars - total_chars
            if remaining <= 0:
                break
            if len(text) > remaining:
                text = text[:remaining]
            total_chars += len(text)
            recency = float(entry.get("recency", 0.0))
            recency_values.append(recency)
            passages.append(
                {
                    "doc_id": entry.get("doc_id"),
                    "score": entry.get("score"),
                    "adjusted_score": entry.get("adjusted_score"),
                    "recency": recency,
                    "text": text,
                    "meta": entry.get("meta", {}),
                    "tags": entry.get("tags", []),
                    "snippet": entry.get("snippet"),
                }
            )

        data["passages"] = passages
        rationale = {
            "why": "Retrieved passages from index",
            "evidence": [
                f"backend={backend}",
                f"returned={len(passages)}",
                f"chars={total_chars}",
            ],
        }
        if recency_days:
            avg_recency = sum(recency_values) / len(recency_values) if recency_values else 0.0
            rationale["evidence"].append(f"fresh_days={recency_days}")
            rationale["evidence"].append(f"avg_recency={avg_recency:.2f}")

        reward = 1.0 if passages else 0.1
        effects = ["has_passages"]
        if recency_days:
            effects.append("fresh")
        log_message = f"Retrieved {len(passages)} passages (backend={backend})."
        if recency_days:
            log_message += f" fresh_days={recency_days}"
        return {
            "ok": True,
            "output": {"passages": passages},
            "logs": [log_message],
            "checks": {},
            "reward": reward,
            "rationale": rationale,
            "effects": effects,
        }


__all__ = ["Retrieve"]
