from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from core.interfaces import Behavior, Context, Result
from core import fetch
from core.index import get_index


class WebRead(Behavior):
    name = "web_read"
    inputs: List[str] = []
    outputs: List[str] = ["passages"]

    def run(self, ctx: Context) -> Result:
        data = ctx.setdefault("data", {})
        url = str(data.get("url") or ctx.get("url") or "").strip()
        if not url:
            return {
                "ok": False,
                "logs": ["web_read: missing url"],
                "reward": 0.0,
                "rewards": {"overall": 0.0},
                "effects": [],
            }

        index_backend = str(data.get("index_backend") or ctx.get("index_backend") or "hnsw").lower()
        index = get_index(backend=index_backend)
        tags_raw = data.get("tags") or ctx.get("tags") or ""
        tags = [tag.strip() for tag in str(tags_raw).split(",") if tag.strip()]
        try:
            text, meta = fetch.fetch_url(url)
        except Exception as exc:
            return {
                "ok": False,
                "logs": [f"web_read: fetch failed ({exc})"],
                "reward": 0.0,
                "rewards": {"overall": 0.0},
                "effects": [],
            }

        canonical_url = meta.get("canonical_url") or meta.get("url") or url
        meta.setdefault("url", canonical_url)
        doc = index.add_document(
            text,
            meta={**meta, "canonical_url": canonical_url},
            tags=tags,
        )

        passages = self._chunk_text(text, canonical_url, doc.doc_id)
        data["passages"] = passages

        rationale = {
            "why": "Fetched and chunked URL content",
            "evidence": [
                f"url={canonical_url}",
                f"passages={len(passages)}",
                f"backend={index_backend}",
            ],
        }
        reward = 1.0 if passages else 0.3
        return {
            "ok": True,
            "output": {"passages": passages},
            "logs": [f"web_read stored document {doc.doc_id}"],
            "reward": reward,
            "rewards": {"overall": reward},
            "effects": ["has_passages"],
            "rationale": rationale,
        }

    def _chunk_text(self, text: str, url: str, doc_id: str, *, chunk_size: int = 800) -> List[Dict[str, Any]]:
        normalized = " ".join(text.split())
        if not normalized:
            return []
        words = normalized.split()
        chunks: List[str] = []
        current: List[str] = []
        count = 0
        for word in words:
            current.append(word)
            count += len(word) + 1
            if count >= chunk_size:
                chunks.append(" ".join(current))
                current = []
                count = 0
        if current:
            chunks.append(" ".join(current))

        passages: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(chunks, start=1):
            passages.append(
                {
                    "doc_id": f"{doc_id}#{idx}",
                    "text": chunk,
                    "meta": {
                        "title": f"{url} (chunk {idx})",
                        "url": url,
                        "canonical_url": url,
                        "source_id": doc_id,
                    },
                }
            )
        return passages


__all__ = ["WebRead"]

