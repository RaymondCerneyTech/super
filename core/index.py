from __future__ import annotations

import datetime as _dt
import os
import json
import math
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _now_utc() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _now_iso() -> str:
    return _now_utc().replace(microsecond=0).isoformat()


def _parse_iso(value: Optional[str]) -> Optional[_dt.datetime]:
    if not value:
        return None
    try:
        return _dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


@dataclass
class Document:
    doc_id: str
    text: str
    meta: Dict[str, str]
    tags: List[str] = field(default_factory=list)
    tokens: List[str] = field(default_factory=list)


class DocumentIndex:
    def __init__(self, backend: str = "tfidf", name: str = "default") -> None:
        self.backend = backend
        self.name = name
        root_env = os.getenv("SUPER_INDEX_DIR")
        base_root = Path(root_env) if root_env else Path("data") / "indexes"
        self.base_path = base_root / backend
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.storage_path = self.base_path / f"{name}.jsonl"
        self.docs: List[Document] = []
        self._load()

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        for line in self.storage_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc = Document(
                doc_id=payload.get("doc_id", str(uuid.uuid4())),
                text=payload.get("text", ""),
                meta=payload.get("meta", {}),
                tags=payload.get("tags", []),
                tokens=payload.get("tokens", []),
            )
            if not doc.tokens:
                doc.tokens = _tokenize(doc.text)
            now_iso = _now_iso()
            if not doc.meta.get("fetched_ts"):
                doc.meta["fetched_ts"] = now_iso
            if not doc.meta.get("published_ts"):
                doc.meta["published_ts"] = doc.meta.get("fetched_ts", now_iso)
            doc.meta.setdefault("ingested_ts", now_iso)
            doc.meta.setdefault("doc_id", doc.doc_id)
            self.docs.append(doc)

    def _persist(self, doc: Document) -> None:
        payload = {
            "doc_id": doc.doc_id,
            "text": doc.text,
            "meta": doc.meta,
            "tags": doc.tags,
            "tokens": doc.tokens,
        }
        with self.storage_path.open("a", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
            handle.write("\n")

    def add_document(self, text: str, meta: Optional[Dict[str, str]] = None, tags: Optional[Iterable[str]] = None) -> Document:
        meta = dict(meta or {})
        now_iso = _now_iso()
        if not meta.get("fetched_ts"):
            meta["fetched_ts"] = now_iso
        if not meta.get("published_ts"):
            meta["published_ts"] = meta["fetched_ts"]
        meta.setdefault("ingested_ts", now_iso)
        tags_list = sorted({str(tag).strip().lower() for tag in (tags or []) if str(tag).strip()})
        doc = Document(doc_id=meta.get("doc_id", str(uuid.uuid4())), text=text, meta=meta, tags=tags_list)
        doc.tokens = _tokenize(text)
        doc.meta["doc_id"] = doc.doc_id
        self.docs.append(doc)
        self._persist(doc)
        return doc

    def size(self) -> int:
        return len(self.docs)

    def _idf(self) -> Dict[str, float]:
        df: Dict[str, int] = {}
        for doc in self.docs:
            seen = set(doc.tokens)
            for token in seen:
                df[token] = df.get(token, 0) + 1
        total_docs = max(1, len(self.docs))
        return {token: math.log(total_docs / (1 + freq)) + 1 for token, freq in df.items()}

    def search(
        self,
        query: str,
        k: int = 10,
        *,
        tags: Optional[Iterable[str]] = None,
        min_score: float = 0.0,
        recency_days: Optional[int] = None,
    ) -> List[Dict[str, object]]:
        if not query.strip():
            return []
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        tag_filter = {str(tag).strip().lower() for tag in (tags or []) if str(tag).strip()}
        idf = self._idf()

        now = _now_utc()
        doc_scores: List[Tuple[float, float, float, Document]] = []
        query_tf: Dict[str, int] = {}
        for token in query_tokens:
            query_tf[token] = query_tf.get(token, 0) + 1

        query_vector = {token: tf * idf.get(token, 1.0) for token, tf in query_tf.items()}
        query_norm = math.sqrt(sum(weight * weight for weight in query_vector.values())) or 1.0

        for doc in self.docs:
            if tag_filter and not tag_filter.intersection(doc.tags):
                continue
            doc_tf: Dict[str, int] = {}
            for token in doc.tokens:
                doc_tf[token] = doc_tf.get(token, 0) + 1
            doc_vector = {token: tf * idf.get(token, 1.0) for token, tf in doc_tf.items()}
            doc_norm = math.sqrt(sum(weight * weight for weight in doc_vector.values())) or 1.0
            dot = sum(query_vector.get(token, 0.0) * weight for token, weight in doc_vector.items())
            score = dot / (query_norm * doc_norm)
            if score < min_score:
                continue
            published_ts = _parse_iso(doc.meta.get("published_ts") or doc.meta.get("fetched_ts"))
            recency = 0.5
            if published_ts:
                age_seconds = (now - published_ts).total_seconds()
                age_days = max(0.0, age_seconds / 86400.0)
                if recency_days and recency_days > 0:
                    recency = max(0.0, min(1.0, 1.0 - (age_days / recency_days)))
                else:
                    recency = max(0.0, min(1.0, 1.0 / (1.0 + age_days / 365.0)))
            weight = (0.5 + 0.5 * recency) if recency_days else (0.7 + 0.3 * recency)
            adjusted_score = score * weight
            doc_scores.append((adjusted_score, score, recency, doc))

        doc_scores.sort(key=lambda item: item[0], reverse=True)
        results: List[Dict[str, object]] = []
        for adjusted_score, raw_score, recency, doc in doc_scores[:k]:
            snippet = doc.text[:500]
            results.append(
                {
                    "doc_id": doc.doc_id,
                    "score": float(raw_score),
                    "adjusted_score": float(adjusted_score),
                    "recency": float(recency),
                    "text": doc.text,
                    "meta": doc.meta,
                    "tags": doc.tags,
                    "snippet": snippet,
                }
            )
        return results


_INDEX_CACHE: Dict[Tuple[str, str], DocumentIndex] = {}


def get_index(backend: str = "tfidf", name: str = "default") -> DocumentIndex:
    key = (backend, name)
    if key not in _INDEX_CACHE:
        _INDEX_CACHE[key] = DocumentIndex(backend=backend, name=name)
    return _INDEX_CACHE[key]


__all__ = ["DocumentIndex", "get_index"]
