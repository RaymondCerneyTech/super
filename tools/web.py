from __future__ import annotations

import html
import io
import re
import urllib.error
import urllib.request
from html.parser import HTMLParser
from typing import Any, Dict, List, Tuple

MAX_WEB_BYTES = 200_000
DEFAULT_TIMEOUT = 10.0
USER_AGENT = "super-cli-tool/1.0 (+https://example.com)"


def _decode_body(body: bytes, headers: Dict[str, str]) -> str:
    encoding = headers.get("content-type", "")
    match = re.search(r"charset=([\\w\\-]+)", encoding, re.IGNORECASE)
    if match:
        charset = match.group(1)
    else:
        charset = "utf-8"
    try:
        return body.decode(charset, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


def web_get(payload: Dict[str, Any]) -> Dict[str, Any]:
    url = str(payload.get("url") or "")
    if not url:
        return {
            "text": "",
            "quality_gain": 0.0,
            "faithfulness": 0.0,
            "notes": "missing url",
        }

    headers = payload.get("headers")
    timeout = float(payload.get("timeout") or DEFAULT_TIMEOUT)
    timeout = max(0.1, min(timeout, DEFAULT_TIMEOUT))

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if isinstance(headers, dict):
        for key, value in headers.items():
            request.add_header(str(key), str(value))

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw_headers = dict(response.headers.items())
            body = response.read(MAX_WEB_BYTES + 1)
            truncated = len(body) > MAX_WEB_BYTES
            body = body[:MAX_WEB_BYTES]
            text = _decode_body(body, raw_headers)
            status = response.getcode() or 200
    except urllib.error.URLError as exc:  # pragma: no cover - network failures
        return {
            "text": "",
            "quality_gain": 0.0,
            "faithfulness": 0.0,
            "notes": f"error: {exc}",
        }

    notes = "ok"
    if truncated:
        notes = "truncated to 200k bytes"
    return {
        "text": text,
        "quality_gain": 0.05,
        "faithfulness": 1.0,
        "notes": notes,
        "status": status,
        "headers": raw_headers,
    }


class _LinkPreservingParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.buffer: List[str] = []
        self._current_link: List[str] = []
        self._href: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, str]]) -> None:
        if tag.lower() == "a":
            href = ""
            for key, value in attrs:
                if key.lower() == "href":
                    href = value or ""
                    break
            self._href.append(href)
            self._current_link.append("")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._current_link:
            text = self._current_link.pop()
            href = self._href.pop() if self._href else ""
            href = href.strip()
            text = text.strip()
            if text and href:
                self.buffer.append(f"[{text}]({href})")
            elif text:
                self.buffer.append(text)

    def handle_data(self, data: str) -> None:
        if self._current_link:
            self._current_link[-1] += data
        else:
            self.buffer.append(data)

    def get_text(self) -> str:
        return "".join(self.buffer)


def html_to_text(payload: Dict[str, Any]) -> Dict[str, Any]:
    raw = payload.get("html") or payload.get("text") or ""
    html_input = str(raw)
    parser = _LinkPreservingParser()
    parser.feed(html_input)
    parser.close()
    text = parser.get_text()
    text = html.unescape(text)
    cleaned = re.sub(r"\\s+", " ", text).strip()
    return {
        "text": cleaned,
        "quality_gain": 0.08,
        "faithfulness": 0.9,
        "notes": "html converted to text",
    }


class _LinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: List[Tuple[str, str]] = []
        self._current: List[str] = []
        self._href: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, str]]) -> None:
        if tag.lower() == "a":
            href = ""
            for key, value in attrs:
                if key.lower() == "href":
                    href = value or ""
                    break
            self._href.append(href)
            self._current.append("")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._current:
            text = self._current.pop().strip()
            href = (self._href.pop() if self._href else "").strip()
            if href:
                self.links.append((href, text))

    def handle_data(self, data: str) -> None:
        if self._current:
            self._current[-1] += data


def extract_links(payload: Dict[str, Any]) -> Dict[str, Any]:
    html_input = str(payload.get("html") or payload.get("text") or "")
    parser = _LinkExtractor()
    parser.feed(html_input)
    parser.close()
    lines = [f"{href}\\t{text}" for href, text in parser.links]
    return {
        "text": "\n".join(lines),
        "quality_gain": 0.04 if lines else 0.0,
        "faithfulness": 1.0,
        "notes": f"{len(lines)} link(s) extracted",
    }


DATE_PATTERN = re.compile(
    r"(\\b\\d{1,2}[/-]\\d{1,2}[/-]\\d{2,4}\\b|\\b\\d{4}[/-]\\d{1,2}[/-]\\d{1,2}\\b|\\b\\w+\\s\\d{1,2},\\s\\d{4}\\b)",
    re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(r"\\b\\d+[\\d,\\.]*\\b")


def extract_facts(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = str(payload.get("text") or "")
    lines = []
    for line in io.StringIO(text):
        if DATE_PATTERN.search(line) or NUMBER_PATTERN.search(line):
            lines.append(line.strip())
    return {
        "text": "\n".join(lines),
        "quality_gain": 0.05 if lines else 0.0,
        "faithfulness": 1.0,
        "notes": f"{len(lines)} fact line(s)",
    }


__all__ = [
    "web_get",
    "html_to_text",
    "extract_links",
    "extract_facts",
]

