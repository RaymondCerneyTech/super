from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup  # type: ignore[import]

from core.interfaces import Behavior, Context, Result


SEARCH_ENDPOINT = "https://duckduckgo.com/html/"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0 Safari/537.36"
    )
}


DEFAULT_WHITELIST = [
    "wikipedia.org",
    "arxiv.org",
    ".edu",
    ".gov",
    ".ac.uk",
]
WHITELIST_PATH = Path("config") / "whitelist_domains.json"
_WHITELIST_CACHE: Optional[List[str]] = None


class WebSearch(Behavior):
    name = "web_search"
    inputs: List[str] = []
    outputs: List[str] = ["search_results"]

    def run(self, ctx: Context) -> Result:
        data = ctx.setdefault("data", {})
        query = str(data.get("query") or ctx.get("query") or data.get("question") or ctx.get("question") or "").strip()
        if not query:
            return {
                "ok": False,
                "logs": ["web_search: missing query"],
                "reward": 0.0,
                "rewards": {"overall": 0.0},
                "effects": [],
            }

        limit = int(data.get("limit") or ctx.get("limit") or 5)
        domain_filter = str(data.get("domain_filter") or ctx.get("domain_filter") or "").strip()
        fresh_days = data.get("fresh_days") or ctx.get("fresh_days")
        freshness = int(fresh_days) if fresh_days else None
        whitelist_only = bool(data.get("whitelist_only") or ctx.get("whitelist_only"))
        whitelist_override = data.get("whitelist_domains") or ctx.get("whitelist_domains")
        whitelist = self._normalize_whitelist(whitelist_override) if whitelist_override else self._load_whitelist()

        try:
            response = requests.post(
                SEARCH_ENDPOINT,
                headers=DEFAULT_HEADERS,
                data={"q": query},
                timeout=10,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            return {
                "ok": False,
                "logs": [f"web_search: request failed ({exc})"],
                "reward": 0.0,
                "rewards": {"overall": 0.0},
                "effects": [],
            }

        soup = BeautifulSoup(response.text, "html.parser")
        results = self._parse_results(soup, limit, domain_filter, whitelist_only, whitelist)
        data["search_results"] = results

        rationale = {
            "why": "Retrieved search results",
            "evidence": [
                f"query={query}",
                f"limit={limit}",
                f"domain_filter={domain_filter or 'none'}",
                f"whitelist={'on' if whitelist_only else 'off'}",
            ],
        }
        reward = 1.0 if results else 0.2
        effects = ["have_search_results"]
        if freshness is not None:
            effects.append("fresh")

        return {
            "ok": True,
            "output": {"search_results": results},
            "logs": [f"web_search retrieved {len(results)} results"],
            "reward": reward,
            "rewards": {"overall": reward},
            "effects": effects,
            "rationale": rationale,
        }

    def _parse_results(
        self,
        soup: BeautifulSoup,
        limit: int,
        domain_filter: str,
        whitelist_only: bool,
        whitelist: List[str],
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for result in soup.select("div.result.results_links_deep"):
            if len(results) >= limit:
                break
            link_tag = result.select_one("a.result__a")
            snippet_tag = result.select_one("a.result__snippet")
            if not link_tag:
                continue
            url = link_tag.get("href") or ""
            if domain_filter and domain_filter not in url:
                continue
            if whitelist_only and not self._matches_whitelist(url, whitelist):
                continue
            title = link_tag.get_text(strip=True)
            snippet = snippet_tag.get_text(" ", strip=True) if snippet_tag else ""
            results.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "fetched_ts": time.time(),
                }
            )
        return results

    def _matches_whitelist(self, url: str, whitelist: List[str]) -> bool:
        host = self._extract_host(url)
        if not host:
            return False
        for entry in whitelist:
            entry_norm = entry.lower().strip()
            if not entry_norm:
                continue
            if entry_norm.startswith("."):
                if host.endswith(entry_norm):
                    return True
            elif host == entry_norm or host.endswith("." + entry_norm):
                return True
        return False

    def _extract_host(self, url: str) -> str:
        try:
            parsed = urlparse(url)
        except Exception:
            return ""
        host = (parsed.hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return host

    def _load_whitelist(self) -> List[str]:
        global _WHITELIST_CACHE
        if _WHITELIST_CACHE is not None:
            return _WHITELIST_CACHE
        whitelist: List[str] = []
        if WHITELIST_PATH.exists():
            try:
                data = json.loads(WHITELIST_PATH.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    whitelist = [str(item).strip() for item in data if str(item).strip()]
            except (OSError, ValueError, TypeError):
                whitelist = []
        if not whitelist:
            whitelist = DEFAULT_WHITELIST.copy()
        _WHITELIST_CACHE = whitelist
        return whitelist

    def _normalize_whitelist(self, raw: object) -> List[str]:
        if isinstance(raw, str):
            items = [part.strip() for part in raw.split(",")]
        elif isinstance(raw, (list, tuple, set)):
            items = [str(part).strip() for part in raw]
        else:
            items = [str(raw).strip()]
        cleaned = [item for item in items if item]
        return cleaned or self._load_whitelist()


__all__ = ["WebSearch"]
