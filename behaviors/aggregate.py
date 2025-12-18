from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse
import re
from typing import Any, Dict, List, Tuple

from core.interfaces import Behavior, Context, Result


class AggregatePassages(Behavior):
    name = "aggregate"
    inputs = ["passages"]
    outputs = ["aggregated_text"]

    def run(self, ctx: Context) -> Result:
        data = ctx.setdefault("data", {})
        data.setdefault("auto_query", [])
        data.setdefault("sc_votes", [])
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

        sources, source_passages, lookup = self._build_sources(passages)
        domain_counts = self._domain_counts(passages)
        unique_domains = [domain for domain in domain_counts if domain]
        data["source_domains"] = sorted(domain_counts.keys())
        data["triangulation_domains"] = [
            {"domain": domain, "count": count} for domain, count in domain_counts.items()
        ]
        source_meta = {src.get("marker"): src for src in sources}
        aggregated_text = ""
        evidence: List[str] = []
        top_passages: List[Dict[str, Any]] = []
        if strategy == "topic_blocks":
            aggregated_text, block_count = self._topic_blocks(passages)
            evidence.append(f"blocks={block_count}")
            top_passages = []
            for src in sources[: min(5, len(sources))]:
                marker = src.get("marker")
                top_passages.append(
                    {
                        "marker": marker,
                        "title": src.get("title", ""),
                        "snippet": source_passages.get(marker or "", "")[:160],
                        "url": src.get("url", ""),
                    }
                )
        else:
            aggregated_text, count, top_passages = self._concatenate(passages, lookup, source_meta)
            evidence.append(f"passages={count}")

        requires_grounding = self._requires_grounding(data)
        triangulation_k = int(
            data.get("triangulation_k")
            or ctx.get("triangulation_k")
            or (2 if requires_grounding else 1)
        )
        triangulation_k = max(1, triangulation_k)
        has_enough_domains = len(unique_domains) >= triangulation_k
        data["triangulation_k"] = triangulation_k
        data["triangulation_met"] = has_enough_domains
        data["needs_more_sources"] = bool(requires_grounding and not has_enough_domains)

        evidence.append(f"domains={len(unique_domains)}")
        if requires_grounding and not has_enough_domains:
            evidence.append("triangulation_pending")

        data["sources"] = sources
        data["source_passages"] = source_passages
        data["has_citable_sources"] = any(src.get("url") for src in sources)
        if top_passages:
            data["top_passages"] = top_passages
        data.setdefault("index_backend", str(data.get("index_backend") or ctx.get("index_backend") or "hnsw"))
        data.setdefault("mpc_mode", data.get("mpc_mode") or "heuristic")

        data["aggregated_text"] = aggregated_text
        if aggregated_text and not data.get("answer") and not self._requires_llm_output(data):
            data["answer"] = aggregated_text

        rationale = {
            "why": f"Aggregated passages using {strategy}",
            "evidence": evidence,
        }
        reward = 1.0 if aggregated_text else 0.2

        effects = ["has_aggregate"]
        if has_enough_domains:
            effects.append("grounded")
        return {
            "ok": True,
            "output": {"aggregated_text": aggregated_text},
            "logs": [f"Aggregated passages with strategy='{strategy}'"],
            "checks": {},
            "reward": reward,
            "rationale": rationale,
            "effects": effects,
        }

    def _requires_grounding(self, data: Dict[str, Any]) -> bool:
        flags = data.get("goal_flags")
        if isinstance(flags, str):
            flags = [frag.strip() for frag in flags.split(",")]
        if isinstance(flags, (list, tuple, set)):
            normalized = {str(flag).strip().lower() for flag in flags}
            return any(flag in {"grounded", "cited"} for flag in normalized)
        return False

    def _domain_counts(self, passages: List[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for passage in passages:
            meta = passage.get("meta", {}) or {}
            domain = self._extract_domain(meta)
            if not domain:
                continue
            counts[domain] = counts.get(domain, 0) + 1
        return counts

    def _extract_domain(self, meta: Dict[str, Any]) -> str:
        if not isinstance(meta, dict):
            return ""
        url = str(meta.get("canonical_url") or meta.get("url") or "").strip()
        host = ""
        if url:
            try:
                parsed = urlparse(url)
                host = (parsed.hostname or "").lower()
            except Exception:
                host = ""
            if host.startswith("www."):
                host = host[4:]
        if host:
            return host
        path = meta.get("path")
        if path:
            try:
                return Path(str(path)).name.lower()
            except Exception:
                return str(path).lower()
        source_id = meta.get("source_id")
        if source_id:
            return str(source_id).lower()
        return ""

    def _requires_llm_output(self, data: Dict[str, Any]) -> bool:
        flags = data.get("goal_flags")
        if isinstance(flags, str):
            flags = [frag.strip() for frag in flags.split(",")]
        if isinstance(flags, (list, tuple, set)):
            normalized = {str(flag).strip().lower() for flag in flags}
            return "llm_output" in normalized
        return False

    def _concatenate(
        self,
        passages: List[Dict[str, object]],
        lookup: Dict[str, str],
        source_meta: Dict[str, Dict[str, Any]],
    ) -> tuple[str, int, List[Dict[str, Any]]]:
        categories = {
            "tactics": [],
            "obstacles": [],
            "resources": [],
            "general": [],
        }
        seen: set[str] = set()
        ordered: List[Dict[str, Any]] = []
        for idx, passage in enumerate(passages, start=1):
            doc_id = str(passage.get("doc_id") or idx)
            if doc_id in seen:
                continue
            seen.add(doc_id)
            meta = passage.get("meta", {}) or {}
            header = (
                meta.get("title")
                or meta.get("url")
                or passage.get("doc_id")
                or f"Source {idx}"
            )
            header = self._sanitize_header(str(header))
            text_body = str(passage.get("text", ""))
            summary = self._summarize(text_body)
            marker = self._resolve_marker(doc_id, header, meta, lookup)
            source_info = source_meta.get(marker, {}) if marker else {}
            ordered.append(
                {
                    "marker": marker,
                    "title": header,
                    "snippet": summary,
                    "url": source_info.get("url", ""),
                }
            )
            text_lower = text_body.lower()
            header_lower = header.lower()
            entry = (marker, summary)
            if any(keyword in header_lower or keyword in text_lower for keyword in ("tactic", "strategy", "approach", "method")):
                categories["tactics"].append(entry)
            elif any(keyword in header_lower or keyword in text_lower for keyword in ("obstacle", "challenge", "barrier", "limitation")):
                categories["obstacles"].append(entry)
            elif any(keyword in header_lower or keyword in text_lower for keyword in ("resource", "collaboration", "roadmap", "project", "dataset", "institute")):
                categories["resources"].append(entry)
            else:
                categories["general"].append(entry)

        sections: List[str] = []

        def emit(title: str, items: List[tuple[str, str]]) -> None:
            if not items:
                return
            sections.append(title)
            for marker, summary in items:
                suffix = f" [{marker}]" if marker else ""
                sections.append(f"- {summary}{suffix}")

        emit("## Current tactics", categories["tactics"])
        emit("## Obstacles and risks", categories["obstacles"])
        emit("## Resources and collaboration", categories["resources"])
        emit("## Additional notes", categories["general"])

        if not sections:
            sections.append("## Sources")
            for marker, summary in categories["general"]:
                suffix = f" [{marker}]" if marker else ""
                sections.append(f"- {summary}{suffix}")

        top_passages = [
            {
                "marker": entry.get("marker"),
                "title": entry.get("title"),
                "snippet": entry.get("snippet"),
                "url": entry.get("url"),
            }
            for entry in ordered[: min(len(ordered), 5)]
        ]
        return "\n\n".join(sections), len(seen), top_passages

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

    def _summarize(self, text: str, max_sentences: int = 2) -> str:
        sentences = [
            sent.strip()
            .replace("–", "-")
            .replace("—", "-")
            .replace("\uFFFD", "-")
            for sent in re.split(r"(?<=[.!?])\s+", text)
            if sent.strip()
        ]
        summary = " ".join(sentences[:max_sentences])
        if not summary:
            summary = text[:200].strip()
        try:
            summary = summary.encode("ascii", "ignore").decode("ascii")
        except Exception:
            pass
        return summary

    def _build_sources(
        self,
        passages: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, str]], Dict[str, str], Dict[str, str]]:
        sources: List[Dict[str, str]] = []
        marker_to_text: Dict[str, str] = {}
        lookup: Dict[str, str] = {}
        seen_keys: set[Tuple[str, str, str]] = set()

        for idx, passage in enumerate(passages, start=1):
            meta = passage.get("meta") or {}
            if not isinstance(meta, dict):
                meta = {}
            raw_title = meta.get("title") or meta.get("url") or meta.get("canonical_url") or meta.get("source_id") or meta.get("path") or "Source"
            title = self._sanitize_header(str(raw_title).strip() or "Source")
            raw_url = meta.get("canonical_url") or meta.get("url") or meta.get("path") or ""
            url = raw_url.strip() if isinstance(raw_url, str) else ""
            if (not url or not isinstance(raw_url, str)) and meta.get("path"):
                try:
                    url = Path(str(meta["path"])).resolve().as_uri()
                except Exception:
                    url = ""
            if url and not url.lower().startswith(("http://", "https://", "file://")):
                url = ""
            doc_id = str(passage.get("doc_id") or idx)
            key = (title, url, doc_id)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            marker = f"S{len(sources) + 1}"
            sources.append(
                {
                    "marker": marker,
                    "title": title,
                    "url": url,
                }
            )
            marker_to_text[marker] = str(passage.get("text", ""))
            lookup[doc_id] = marker
            lookup[f"{title.lower()}|{url.lower()}"] = marker
            if url:
                lookup[f"|{url.lower()}"] = marker

        return sources, marker_to_text, lookup

    def _resolve_marker(self, doc_id: str, title: str, meta: Any, lookup: Dict[str, str]) -> str:
        marker = lookup.get(doc_id, "")
        url = ""
        if isinstance(meta, dict):
            raw_url = meta.get("canonical_url") or meta.get("url") or ""
            url = str(raw_url).strip().lower()
        if url:
            marker = marker or lookup.get(f"{title.lower()}|{url}") or lookup.get(f"|{url}")
        if not marker:
            marker = lookup.get(f"{title.lower()}|", "")
        return marker

    def _sanitize_header(self, header: str) -> str:
        try:
            cleaned = header.encode("ascii", "ignore").decode("ascii")
        except Exception:
            cleaned = header
        return cleaned


__all__ = ["AggregatePassages"]
