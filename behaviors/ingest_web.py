from __future__ import annotations

from urllib.parse import urlparse
from typing import Dict, List, Optional

from core import fetch as fetchers
from core.interfaces import Behavior, Context, Result
from core.index import get_index


class IngestWeb(Behavior):
    name = "ingest_web"
    inputs: List[str] = []
    outputs: List[str] = []

    def run(self, ctx: Context) -> Result:
        data = ctx.setdefault("data", {})
        url = data.get("url") or ctx.get("url")
        rss = data.get("rss") or ctx.get("rss")
        path = data.get("path") or ctx.get("path")
        tags_raw = data.get("tags") or ctx.get("tags") or ""
        tag_list = self._normalize_tags(tags_raw)
        backend = (data.get("index_backend") or ctx.get("index_backend") or "tfidf").lower()
        limit = int(data.get("limit") or ctx.get("limit") or 15)
        lang_any = bool(data.get("lang_any") or ctx.get("lang_any"))
        logs: List[str] = []
        ingested: List[Dict[str, str]] = []
        skipped: List[str] = []

        if sum(bool(value) for value in (url, rss, path)) != 1:
            return {
                "ok": False,
                "logs": ["Provide exactly one of url, rss, or path"],
                "checks": {},
                "reward": 0.0,
                "rationale": {"why": "Invalid ingest parameters", "evidence": []},
                "effects": [],
            }

        index = get_index(backend=backend)

        if path:
            try:
                text = self._read_file(path)
            except OSError as err:
                logs.append(f"Failed to read {path}: {err}")
                return {
                    "ok": False,
                    "logs": logs,
                    "checks": {},
                    "reward": 0.0,
                    "rationale": {"why": "Unable to read local file", "evidence": logs},
                    "effects": [],
                }
            if not text.strip():
                logs.append(f"Empty file: {path}")
            elif lang_any or fetchers.is_probably_english(text):
                meta = {"source": "file", "path": str(path)}
                doc = index.add_document(text, meta, tags=tag_list)
                ingested.append(self._record_from_doc(doc, meta, tag_list))
            else:
                skipped.append(path)
        elif rss:
            items = fetchers.fetch_rss(rss, limit=limit)
            logs.append(f"Fetched {len(items)} feed items from {rss}")
            for item in items:
                link = item.get("link") or ""
                if not link:
                    continue
                text, meta = fetchers.fetch_url(link)
                meta.update({"source": "rss", "feed_url": rss, "title": item.get("title", "")})
                if not text.strip():
                    skipped.append(link)
                    continue
                if not lang_any and not fetchers.is_probably_english(text):
                    skipped.append(f"{link} (language filter)")
                    continue
                combined_tags = tag_list + self._host_tags(link)
                doc = index.add_document(text, meta, tags=combined_tags)
                ingested.append(self._record_from_doc(doc, meta, combined_tags))
        else:  # url flow
            text, meta = fetchers.fetch_url(url)  # type: ignore[arg-type]
            meta.setdefault("source", "url")
            if not text.strip():
                logs.append(f"No text extracted from {url}")
            elif lang_any or fetchers.is_probably_english(text):
                combined_tags = tag_list + self._host_tags(url)  # type: ignore[arg-type]
                doc = index.add_document(text, meta, tags=combined_tags)
                ingested.append(self._record_from_doc(doc, meta, combined_tags))
            else:
                skipped.append(f"{url} (language filter)")

        previous_log = data.get("ingest_log")
        if isinstance(previous_log, list):
            data["ingest_log"] = previous_log + ingested
        else:
            data["ingest_log"] = list(ingested)
        rationale = {
            "why": "Web documents ingested",
            "evidence": [
                f"docs_added={len(ingested)}",
                f"backend={backend}",
                f"skipped={len(skipped)}",
            ],
        }
        logs.extend(f"Skipped: {item}" for item in skipped)
        effects = ["has_knowledge"] if ingested else []
        reward = 1.0 if ingested else 0.2
        return {
            "ok": True,
            "output": {"ingested": ingested, "skipped": skipped},
            "logs": logs,
            "checks": {},
            "reward": reward,
            "rationale": rationale,
            "effects": effects,
        }

    def _normalize_tags(self, tags_raw: object) -> List[str]:
        if not tags_raw:
            return []
        if isinstance(tags_raw, str):
            items = tags_raw.split(",")
        elif isinstance(tags_raw, list):
            items = tags_raw
        else:
            items = [str(tags_raw)]
        return [tag.strip().lower() for tag in items if str(tag).strip()]

    def _host_tags(self, url: str) -> List[str]:
        try:
            parsed = urlparse(url)
            host = parsed.hostname or ""
            parts = host.split(".")
            return [host] + parts[-2:] if host else []
        except Exception:
            return []

    def _read_file(self, path: str) -> str:
        file_path = Path(path)
        return file_path.read_text(encoding="utf-8")

    def _record_from_doc(self, doc, meta: Dict[str, str], tags: List[str]) -> Dict[str, str]:
        record = {
            "doc_id": getattr(doc, "doc_id", ""),
            "url": meta.get("url") or meta.get("path") or "",
            "title": meta.get("title") or meta.get("url") or meta.get("path") or meta.get("source", ""),
            "parser": meta.get("parser", ""),
            "tags": ",".join(tags),
        }
        if meta.get("published_ts"):
            record["published_ts"] = meta["published_ts"]
        if meta.get("fetched_ts"):
            record["fetched_ts"] = meta["fetched_ts"]
        return record


__all__ = ["IngestWeb"]
