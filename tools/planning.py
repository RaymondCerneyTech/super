from __future__ import annotations

import datetime as dt
import re
from typing import Any, Dict, Iterable, List, Sequence


def causal_dot_builder(payload: Dict[str, Any]) -> Dict[str, Any]:
    edges = payload.get("edges") or payload.get("text") or []
    if isinstance(edges, str):
        edges = [part.strip() for part in edges.splitlines() if part.strip()]
    normalized: List[Sequence[str]] = []
    for entry in edges:
        if isinstance(entry, str):
            if "->" in entry:
                src, dst = entry.split("->", 1)
                normalized.append((src.strip(), dst.strip()))
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            normalized.append((str(entry[0]).strip(), str(entry[1]).strip()))
    lines = ["digraph G {"]
    for src, dst in normalized:
        if not src or not dst:
            continue
        lines.append(f'  "{src}" -> "{dst}";')
    lines.append("}")
    dot = "\n".join(lines)
    return {
        "text": dot,
        "quality_gain": 0.05 if normalized else 0.0,
        "faithfulness": 1.0,
        "notes": f"{len(normalized)} edge(s) encoded",
    }


RELATIVE_KEYWORDS = {
    "today": 0,
    "tomorrow": 1,
    "yesterday": -1,
    "in two days": 2,
    "in three days": 3,
}

WEEKDAY_INDEX = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def _parse_week_reference(text: str, today: dt.date) -> List[str]:
    text_lower = text.lower()
    matches = []
    for word, index in WEEKDAY_INDEX.items():
        if word in text_lower:
            matches.append((text_lower.index(word), word, index))
    if not matches:
        return []
    results = []
    for _, word, target_idx in matches:
        delta = target_idx - today.weekday()
        if "next" in text_lower:
            delta += 7 if delta <= 0 else 0
        elif "last" in text_lower:
            delta -= 7 if delta >= 0 else 0
        date = today + dt.timedelta(days=delta)
        results.append(date.isoformat())
    return results


DATE_PATTERN = re.compile(
    r"(?P<month>\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*)\\s+(?P<day>\\d{1,2})(?:,\\s*(?P<year>\\d{4}))?",
    re.IGNORECASE,
)


def _normalize_dates(text: str, today: dt.date) -> List[str]:
    results: List[str] = []
    for keyword, offset in RELATIVE_KEYWORDS.items():
        if keyword in text.lower():
            results.append((today + dt.timedelta(days=offset)).isoformat())
    results.extend(_parse_week_reference(text, today))
    for match in DATE_PATTERN.finditer(text):
        month_text = match.group("month").lower()
        month_map = {
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "may": 5,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "sept": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12,
        }
        month = month_map.get(month_text[:3], 1)
        day = int(match.group("day"))
        year = int(match.group("year") or today.year)
        try:
            results.append(dt.date(year, month, day).isoformat())
        except ValueError:
            continue
    return results


def timeline_normalize(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = str(payload.get("text") or "")
    today = payload.get("today")
    if isinstance(today, str):
        today = dt.date.fromisoformat(today)
    elif isinstance(today, dt.date):
        today = today
    else:
        today = dt.date.today()

    dates = _normalize_dates(text, today)
    notes = "extracted dates"
    output_text = ""
    if len(dates) == 1:
        output_text = dates[0]
    elif len(dates) >= 2:
        output_text = f"start={dates[0]}, end={dates[-1]}"
    else:
        notes = "no recognizable dates"
    return {
        "text": output_text,
        "quality_gain": 0.05 if dates else 0.0,
        "faithfulness": 1.0,
        "notes": notes,
        "dates": dates,
    }


__all__ = ["causal_dot_builder", "timeline_normalize"]

