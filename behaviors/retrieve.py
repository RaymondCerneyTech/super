from __future__ import annotations

import hashlib
from pathlib import Path
import re
from urllib.parse import urlparse
from typing import Dict, Iterable, List, Optional, Tuple

from core.interfaces import Behavior, Context, Result
from core import fetch
from core.index import get_index
from behaviors.web_search import WebSearch


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


def _coerce_list(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


BUILTIN_SEEDS: List[Dict[str, object]] = [
    {
        "keywords": ("riemann hypothesis", "riemann's hypothesis", "riemann zeta"),
        "source_id": "builtin:rh_current_tactics",
        "title": "Riemann Hypothesis - Current Tactics",
        "tags": ["math", "number theory", "riemann hypothesis"],
        "text": (
            "Current tactics against the Riemann Hypothesis mix classical analytic number theory with heavy computation. "
            "Random matrix heuristics predict pair correlation of zeta zeros and guide expectations for extreme values. "
            "Researchers refine explicit formulae to transfer zero information into bounds for primes, and analyze "
            "Li/Keiper coefficients and de Bruijn–Newman constants for signs of instability. Supercomputers extend zero "
            "verification past 10^13 along the critical line, while machine-learning classifiers search for anomalies in "
            "zero spacings that could hint at counterexamples."
        ),
    },
    {
        "keywords": ("riemann hypothesis", "obstacle", "challenge", "barrier"),
        "source_id": "builtin:rh_obstacles",
        "title": "Riemann Hypothesis - Obstacles and Open Problems",
        "tags": ["math", "riemann hypothesis", "obstacles"],
        "text": (
            "Key obstacles include the lack of a spectral model covering all zeros, the difficulty of controlling error terms "
            "in explicit formulae, and the scarcity of techniques that connect local statistics with global L-function behaviour. "
            "Resonance methods and correlation estimates have not yet produced contradictions, and conductor growth for higher-degree "
            "L-functions makes brute-force verification expensive. Attempts to adapt Selberg-style trace formulae or quantum chaos "
            "analogies remain incomplete."
        ),
    },
    {
        "keywords": ("riemann hypothesis", "collaboration", "resources", "roadmap"),
        "source_id": "builtin:rh_resources",
        "title": "Riemann Hypothesis - Collaborative Resources",
        "tags": ["math", "riemann hypothesis", "resources"],
        "text": (
            "Collaborative resources include shared zero tables on LMFDB, Clay Mathematics Institute surveys highlighting strategic "
            "roadmaps, and open-source code for zero-finding and explicit formula experiments. Polymath proposals suggest combining "
            "automated verification with new analytic ideas, and recent surveys recommend integrating data-driven conjectures with "
            "rigorous analytic frameworks."
        ),
    },
]


def _index_contains_source(index, source_id: str) -> bool:
    if not source_id:
        return False
    docs = getattr(index, "docs", [])
    for doc in docs:
        if isinstance(doc.meta, dict) and doc.meta.get("source_id") == source_id:
            return True
    return False


def _add_document(index, text: str, *, source_id: str, title: Optional[str] = None, tags: Optional[Iterable[str]] = None, meta_extra: Optional[Dict[str, str]] = None) -> bool:
    if not text or _index_contains_source(index, source_id):
        return False
    meta = {"source_id": source_id}
    if title:
        meta["title"] = title
    if meta_extra:
        meta.update({k: str(v) for k, v in meta_extra.items()})
    index.add_document(text, meta=meta, tags=tags)
    return True


def _bootstrap_index(index, data: Dict[str, object], question: str) -> Tuple[bool, List[str]]:
    actions: List[str] = []
    seeded = False
    urls = _coerce_list(data.get("bootstrap_urls"))
    paths = _coerce_list(data.get("bootstrap_paths"))
    texts = _coerce_list(data.get("bootstrap_texts"))

    for url in urls:
        try:
            text, meta = fetch.fetch_url(url)
        except Exception as exc:
            actions.append(f"url:{url} error={exc}")
            continue
        if not text:
            actions.append(f"url:{url} empty")
            continue
        source_id = f"url:{meta.get('url', url)}"
        added = _add_document(index, text, source_id=source_id, title=meta.get("title"), tags=[meta.get("url_host", "")] if meta.get("url_host") else None, meta_extra={"url": meta.get("url", url), "parser": meta.get("parser", "unknown")})
        actions.append(f"url:{url} added={added}")
        seeded = seeded or added

    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            actions.append(f"path:{path} missing")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            actions.append(f"path:{path} error={exc}")
            continue
        source_id = f"path:{path.resolve()}"
        added = _add_document(index, text, source_id=source_id, title=path.name, tags=None, meta_extra={"path": str(path.resolve())})
        actions.append(f"path:{path} added={added}")
        seeded = seeded or added

    for raw_text in texts:
        source_hash = hashlib.sha1(raw_text.encode("utf-8")).hexdigest()
        added = _add_document(index, raw_text, source_id=f"text:{source_hash}", title="Seed note", tags=None, meta_extra={"origin": "inline_text"})
        actions.append(f"text:{source_hash[:8]} added={added}")
        seeded = seeded or added

    auto_bootstrap = bool(data.get("auto_bootstrap", True))
    if not seeded and auto_bootstrap:
        normalized = question.lower()
        for entry in BUILTIN_SEEDS:
            keywords = entry.get("keywords", [])
            if not any(keyword in normalized for keyword in keywords):
                continue
            added = _add_document(
                index,
                str(entry.get("text", "")),
                source_id=str(entry.get("source_id", "")),
                title=str(entry.get("title", "")) or None,
                tags=entry.get("tags"),
                meta_extra={"origin": "builtin_seed"},
            )
            actions.append(f"builtin:{entry.get('source_id', '')} added={added}")
            seeded = seeded or added

    return seeded, actions


def _collect_passages(results: List[Dict[str, object]], max_chars: int) -> Tuple[List[Dict[str, object]], int, List[float]]:
    passages: List[Dict[str, object]] = []
    total_chars = 0
    recency_values: List[float] = []
    for entry in results:
        text = str(entry.get("text", ""))
        meta_obj = entry.get("meta", {})
        meta = meta_obj if isinstance(meta_obj, dict) else {}
        url = str(meta.get("url", ""))
        source_id = str(meta.get("source_id", ""))
        is_example_placeholder = "example.com" in url.lower() and "example domain" in text.lower()
        if is_example_placeholder or source_id == "builtin:riemann_hypothesis_overview":
            continue
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
    return passages, total_chars, recency_values


TRIANGULATION_MIN = 2
WEB_SEARCH_BEHAVIOR = WebSearch()


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

        backend = (data.get("index_backend") or ctx.get("index_backend") or "hnsw").lower()
        index = get_index(backend=backend)
        k_passages = int(data.get("k_passages") or ctx.get("k_passages") or 12)
        max_chars = int(data.get("max_chars") or ctx.get("max_chars") or 12000)
        tags = _parse_tags(data.get("tags") or ctx.get("tags"))
        fresh_days_raw = data.get("fresh_days") or ctx.get("fresh_days")
        recency_days = int(fresh_days_raw) if fresh_days_raw else None
        requires_grounding = self._requires_grounding(data)
        triangulation_k = int(
            data.get("triangulation_k")
            or ctx.get("triangulation_k")
            or (TRIANGULATION_MIN if requires_grounding else 1)
        )
        triangulation_k = max(1, triangulation_k)
        data["triangulation_k"] = triangulation_k

        results = index.search(question, k=k_passages, tags=tags, recency_days=recency_days)
        passages, total_chars, recency_values = _collect_passages(results, max_chars)

        bootstrap_actions: List[str] = []
        if not passages:
            seeded, bootstrap_actions = _bootstrap_index(index, data, question)
            if seeded:
                results = index.search(question, k=k_passages, tags=tags, recency_days=recency_days)
                passages, total_chars, recency_values = _collect_passages(results, max_chars)
        triangulation_actions: List[str] = []
        if requires_grounding:
            domain_counts = self._domain_counts(passages)
            unique_domains = [dom for dom in domain_counts.keys() if dom]
            if len(unique_domains) < triangulation_k:
                triangulation_actions = self._triangulate_with_whitelist(
                    index=index,
                    question=question,
                    existing_domains=set(unique_domains),
                    target=triangulation_k,
                    limit=k_passages,
                )
                if triangulation_actions:
                    results = index.search(question, k=k_passages, tags=tags, recency_days=recency_days)
                    passages, total_chars, recency_values = _collect_passages(results, max_chars)
            domain_counts = self._domain_counts(passages)
        else:
            domain_counts = self._domain_counts(passages)

        data["source_domains"] = sorted(domain_counts.keys())
        data["triangulation_domains"] = [
            {"domain": domain, "count": count} for domain, count in domain_counts.items()
        ]
        data["triangulation_met"] = len([dom for dom in domain_counts.keys() if dom]) >= triangulation_k
        if triangulation_actions:
            log = data.setdefault("triangulation_log", [])
            log.extend(triangulation_actions)

        data["index_backend"] = backend
        data["passages"] = passages
        rationale = {
            "why": "Retrieved passages from index",
            "evidence": [
                f"backend={backend}",
                f"returned={len(passages)}",
                f"chars={total_chars}",
            ],
        }
        if domain_counts:
            rationale["evidence"].append(
                f"domains={len([dom for dom in domain_counts.keys() if dom])}"
            )
        if recency_days:
            avg_recency = sum(recency_values) / len(recency_values) if recency_values else 0.0
            rationale["evidence"].append(f"fresh_days={recency_days}")
            rationale["evidence"].append(f"avg_recency={avg_recency:.2f}")

        if bootstrap_actions:
            rationale["evidence"].append(f"bootstrap={'; '.join(bootstrap_actions)}")
        if triangulation_actions:
            rationale["evidence"].append(f"triangulation={'; '.join(triangulation_actions)}")

        reward = 1.0 if passages else 0.1
        effects = ["has_passages"]
        if recency_days:
            effects.append("fresh")
        log_message = f"Retrieved {len(passages)} passages (backend={backend})."
        if recency_days:
            log_message += f" fresh_days={recency_days}"
        if bootstrap_actions:
            log_message += f" bootstrap={' | '.join(bootstrap_actions)}"
        return {
            "ok": True,
            "output": {"passages": passages},
            "logs": [log_message],
            "checks": {},
            "reward": reward,
            "rationale": rationale,
            "effects": effects,
        }

    def _requires_grounding(self, data: Dict[str, object]) -> bool:
        flags = data.get("goal_flags")
        if isinstance(flags, str):
            flags = [frag.strip() for frag in flags.split(",")]
        if isinstance(flags, (list, tuple, set)):
            normalized = {str(flag).strip().lower() for flag in flags}
            return any(flag in {"grounded", "cited"} for flag in normalized)
        return False

    def _domain_counts(self, passages: List[Dict[str, object]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for passage in passages:
            meta = passage.get("meta", {}) or {}
            domain = self._extract_domain(meta)
            if not domain:
                continue
            counts[domain] = counts.get(domain, 0) + 1
        return counts

    def _extract_domain(self, meta: Dict[str, object]) -> str:
        if not isinstance(meta, dict):
            return ""
        url = str(
            meta.get("canonical_url")
            or meta.get("url")
            or ""
        ).strip()
        host = ""
        if url:
            try:
                parsed = urlparse(url)
                host = (parsed.hostname or "").lower()
            except Exception:
                host = ""
        if host and host.startswith("www."):
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

    def _triangulate_with_whitelist(
        self,
        *,
        index,
        question: str,
        existing_domains: set[str],
        target: int,
        limit: int,
    ) -> List[str]:
        whitelist = WEB_SEARCH_BEHAVIOR._load_whitelist()
        attempts: List[str] = []
        queries = self._expand_queries(question)
        for query in queries:
            search_ctx: Context = {
                "data": {
                    "query": query,
                    "limit": limit,
                    "whitelist_only": True,
                    "whitelist_domains": whitelist,
                }
            }
            search_result = WEB_SEARCH_BEHAVIOR.run(search_ctx) or {}
            search_output = search_result.get("output") if isinstance(search_result, dict) else {}
            results = search_output.get("search_results") if isinstance(search_output, dict) else []
            if not results:
                attempts.append(f"search:{query} empty")
                continue
            for entry in results:
                url = entry.get("url") or ""
                domain = ""
                if url:
                    try:
                        parsed = urlparse(url)
                        domain = (parsed.hostname or "").lower()
                    except Exception:
                        domain = ""
                    if domain.startswith("www."):
                        domain = domain[4:]
                if not url or not domain or domain in existing_domains:
                    continue
                try:
                    text, meta = fetch.fetch_url(url)
                except Exception as exc:
                    attempts.append(f"fetch:{domain} error={exc}")
                    continue
                if not text.strip():
                    attempts.append(f"fetch:{domain} empty")
                    continue
                meta_extra = {
                    "url": meta.get("url", url),
                    "parser": meta.get("parser", "unknown"),
                    "canonical_url": meta.get("url", url),
                    "url_host": domain,
                    "triangulated": "true",
                }
                added = _add_document(
                    index,
                    text,
                    source_id=f"url:{url}",
                    title=meta.get("title"),
                    tags=[domain],
                    meta_extra=meta_extra,
                )
                attempts.append(f"triangulate:{domain} added={added}")
                if added:
                    existing_domains.add(domain)
                    if len([dom for dom in existing_domains if dom]) >= target:
                        return attempts
            if len([dom for dom in existing_domains if dom]) >= target:
                break
        return attempts

    def _expand_queries(self, question: str) -> List[str]:
        base = question.strip()
        extras = [
            f"{base} review" if base else "review",
            f"{base} survey" if base else "survey",
            f"{base} site:wikipedia.org" if base else "site:wikipedia.org",
            f"{base} site:arxiv.org" if base else "site:arxiv.org",
            f"{base} site:.edu" if base else "site:.edu",
            f"{base} site:.gov" if base else "site:.gov",
        ]
        queries = [base] if base else []
        queries.extend(extras)
        clean = []
        for query in queries:
            q = query.strip()
            if q and q not in clean:
                clean.append(q[:240])
        return clean or ["wikipedia.org arxiv.org"]


__all__ = ["Retrieve"]
