from __future__ import annotations

import datetime as _dt
import io
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

USER_AGENT = "SuperAI/0.1 (+https://example.com)"


def _now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def is_probably_english(text: str, min_ratio: float = 0.6, min_length: int = 64) -> bool:
    if not text:
        return False
    sample = text[:2000]
    letters = sum(1 for ch in sample if ch.isalpha())
    ascii_letters = sum(1 for ch in sample if "LATIN" in unicodedata.name(ch, ""))
    if letters == 0:
        return False
    ratio = ascii_letters / max(1, letters)
    if len(sample) < min_length:
        return ratio >= (min_ratio + 0.1)
    return ratio >= min_ratio


def _get_requests():
    try:
        import requests  # type: ignore

        return requests
    except Exception:
        return None


def _fetch_with_requests(url: str, timeout: int = 15):
    requests = _get_requests()
    if not requests:
        raise RuntimeError("requests not available")
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    return resp.content, resp.headers.get("content-type") or "", resp.url


def _fetch_with_urllib(url: str, timeout: int = 15):
    from urllib import request

    req = request.Request(url, headers={"User-Agent": USER_AGENT})
    with request.urlopen(req, timeout=timeout) as resp:
        content_type = resp.headers.get("Content-Type", "")
        data = resp.read()
        final_url = resp.geturl()
    return data, content_type, final_url


def _detect_content_type(headers_ct: str, url: str) -> str:
    if headers_ct:
        return headers_ct.lower()
    if url.endswith(".pdf"):
        return "application/pdf"
    if url.endswith(".html") or url.endswith(".htm"):
        return "text/html"
    if url.endswith(".txt"):
        return "text/plain"
    return "application/octet-stream"


def _clean_html(raw_html: bytes) -> str:
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except Exception:
        return ""

    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style", "nav", "noscript", "footer", "header", "aside"]):
        tag.decompose()
    content_root = _select_content_root(soup)
    pieces: List[str] = []
    allowed_tags = ["h1", "h2", "h3", "h4", "p", "li"]
    for element in content_root.find_all(allowed_tags):
        text = element.get_text(" ", strip=True)
        if not text:
            continue
        if _should_skip_line(text):
            continue
        pieces.append(text)
    text = "\n".join(_deduplicate_lines(pieces))
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()

def _select_content_root(soup) -> Any:
    article = soup.find("article")
    if article and len(article.get_text(strip=True)) >= 200:
        return article
    main = soup.find("main")
    if main and len(main.get_text(strip=True)) >= 200:
        return main
    candidates = []
    for node in soup.find_all(["section", "div"], recursive=True):
        if not hasattr(node, "get_text"):
            continue
        if node.name in {"nav", "footer", "header", "aside"}:
            continue
        text = node.get_text(" ", strip=True)
        length = len(text)
        if length < 200:
            continue
        candidates.append((length, node))
    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]
    return soup


def _deduplicate_lines(lines: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for line in lines:
        normalized = line.strip()
        if not normalized:
            continue
        if normalized.lower() in seen:
            continue
        seen.add(normalized.lower())
        result.append(normalized)
    return result


COOKIE_STRINGS = [
    "we use cookies",
    "accept cookies",
    "privacy policy",
    "cookie policy",
    "we need your consent",
    "subscribe to",
    "follow us on",
]


def _should_skip_line(line: str) -> bool:
    normalized = line.strip().lower()
    if len(normalized) <= 3:
        return True
    for phrase in COOKIE_STRINGS:
        if phrase in normalized:
            return True
    return False


def _clean_pdf(raw_pdf: bytes) -> str:
    try:
        from pdfminer.high_level import extract_text  # type: ignore
    except Exception:
        return ""

    try:
        with io.BytesIO(raw_pdf) as buffer:
            return extract_text(buffer) or ""
    except Exception:
        return ""


def _decode_text(raw: bytes, encoding_hint: Optional[str]) -> str:
    encodings = []
    if encoding_hint:
        m = re.search(r"charset=([\w-]+)", encoding_hint)
        if m:
            encodings.append(m.group(1))
    encodings.extend(["utf-8", "latin-1"])
    for enc in encodings:
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("utf-8", errors="ignore")


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_url(url: str) -> Tuple[str, Dict[str, str]]:
    """
    Fetch remote resource, returning normalized text plus metadata.
    """
    fetch_error = None
    final_url = url
    headers_ct = ""
    raw: bytes = b""
    try:
        raw, headers_ct, final_url = _fetch_with_requests(url)
    except Exception as exc:
        fetch_error = str(exc)
        try:
            raw, headers_ct, final_url = _fetch_with_urllib(url)
            fetch_error = None
        except Exception as inner_exc:
            fetch_error = f"requests+urllib failed: {inner_exc}"

    content_type = _detect_content_type(headers_ct, final_url)
    parser = "unknown"
    text_output = ""

    if raw:
        if "text/html" in content_type:
            parser = "html"
            text_output = _clean_html(raw)
        elif "pdf" in content_type:
            parser = "pdf"
            text_output = _clean_pdf(raw)
        elif content_type.startswith("text/"):
            parser = "text"
            text_output = _decode_text(raw, headers_ct)
        else:
            parser = "binary"
            text_output = ""

    if not text_output and raw and parser == "binary":
        parser = "text"
        text_output = _decode_text(raw, headers_ct)

    text_output = _normalize_whitespace(text_output or "")

    meta: Dict[str, str] = {
        "url": final_url,
        "requested_url": url,
        "content_type": content_type,
        "parser": parser,
        "fetched_ts": _now_iso(),
    }
    if fetch_error:
        meta["error"] = fetch_error
    if not text_output and not fetch_error:
        meta["warning"] = "No textual content extracted"
    return text_output, meta


def fetch_rss(feed_url: str, limit: int = 15) -> List[Dict[str, str]]:
    try:
        import feedparser  # type: ignore
    except Exception:
        return []

    feed = feedparser.parse(feed_url)
    items: List[Dict[str, str]] = []
    for entry in feed.entries[:limit]:
        link = getattr(entry, "link", "")
        title = getattr(entry, "title", "")
        published = ""
        if getattr(entry, "published_parsed", None):
            published = _dt.datetime(*entry.published_parsed[:6]).isoformat() + "Z"
        elif getattr(entry, "updated_parsed", None):
            published = _dt.datetime(*entry.updated_parsed[:6]).isoformat() + "Z"
        items.append(
            {
                "title": title,
                "link": link,
                "published_ts": published,
            }
        )
    return items


__all__ = ["fetch_url", "fetch_rss", "is_probably_english"]
