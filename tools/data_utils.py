from __future__ import annotations

import csv
import io
import json
import re
from typing import Any, Dict, Iterable, List, Sequence, Union

MAX_TEXT = 200_000


def table_detect(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = str(payload.get("text") or "")
    if not text:
        return {
            "text": "",
            "quality_gain": 0.0,
            "faithfulness": 0.0,
            "notes": "no text provided",
        }

    sample = text.strip().splitlines()
    if not sample:
        return {
            "text": "",
            "quality_gain": 0.0,
            "faithfulness": 0.0,
            "notes": "no lines detected",
        }

    delimiters = [",", "\t", "|", ";"]
    best_delim = ","
    best_score = -1
    for delim in delimiters:
        score = sum(line.count(delim) for line in sample[:10])
        if score > best_score:
            best_score = score
            best_delim = delim

    rows = [re.split(rf"\\s*{re.escape(best_delim)}\\s*", line.strip()) for line in sample if line.strip()]
    if not rows:
        rows = [line.split() for line in sample]

    header = rows[0]
    if any(col.isdigit() for col in header):
        header = [f"col_{idx+1}" for idx in range(len(rows[0]))]
        rows = [header] + rows

    output = io.StringIO()
    writer = csv.writer(output)
    for row in rows:
        writer.writerow(row)
    csv_text = output.getvalue()
    return {
        "text": csv_text,
        "quality_gain": 0.1,
        "faithfulness": 0.9,
        "notes": f"detected {len(rows)} row(s) using '{best_delim}' delimiter",
    }


def _numeric_stats(values: Sequence[float]) -> str:
    if not values:
        return ""
    avg = sum(values) / len(values)
    return f"min={min(values):.3f}, max={max(values):.3f}, mean={avg:.3f}"


def csv_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    csv_text = str(payload.get("csv") or payload.get("text") or "")
    if not csv_text:
        return {
            "text": "",
            "quality_gain": 0.0,
            "faithfulness": 0.0,
            "notes": "no csv text provided",
        }
    reader = csv.reader(io.StringIO(csv_text))
    try:
        header = next(reader)
    except StopIteration:
        return {
            "text": "",
            "quality_gain": 0.0,
            "faithfulness": 0.0,
            "notes": "empty csv",
        }
    rows = list(reader)
    column_stats: List[str] = []
    for idx, name in enumerate(header):
        column = []
        numeric_values: List[float] = []
        for row in rows:
            if idx < len(row):
                value = row[idx].strip()
                column.append(value)
                try:
                    numeric_values.append(float(value.replace(",", "")))
                except ValueError:
                    continue
        detail = f"{name}: {len(column)} values"
        numeric_summary = _numeric_stats(numeric_values)
        if numeric_summary:
            detail += f" ({numeric_summary})"
        column_stats.append(detail)

    summary = [
        f"columns: {', '.join(header)}",
        f"rows: {len(rows)}",
    ]
    summary.extend(column_stats)
    return {
        "text": "\n".join(summary),
        "quality_gain": 0.08,
        "faithfulness": 1.0,
        "notes": "csv summary computed",
    }


def _load_json_source(payload: Dict[str, Any]) -> Any:
    if "json_obj" in payload:
        return payload["json_obj"]
    if "json_text" in payload or "text" in payload:
        source = payload.get("json_text", payload.get("text"))
        try:
            return json.loads(source)
        except json.JSONDecodeError:
            raise ValueError("invalid json input")
    raise ValueError("json_obj or json_text required")


def _tokenize_path(path: str) -> List[Union[str, int]]:
    tokens: List[Union[str, int]] = []
    for segment in path.split("."):
        segment = segment.strip()
        if not segment:
            continue
        while segment:
            bracket = segment.find("[")
            if bracket == -1:
                tokens.append(segment)
                break
            name = segment[:bracket]
            if name:
                tokens.append(name)
            match = re.match(r"\[(\d+)\](.*)", segment[bracket:])
            if not match:
                raise ValueError(f"invalid path segment '{segment}'")
            tokens.append(int(match.group(1)))
            segment = match.group(2)
    return tokens


def _traverse_path(obj: Any, tokens: Iterable[Union[str, int]]) -> Any:
    current = obj
    for token in tokens:
        if isinstance(token, int):
            if not isinstance(current, (list, tuple)):
                raise TypeError(f"cannot index into {type(current).__name__}")
            current = current[token]
        else:
            if not isinstance(current, dict):
                raise TypeError(f"cannot access key '{token}' on {type(current).__name__}")
            current = current[token]
    return current


def json_query(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        source = _load_json_source(payload)
    except ValueError as exc:
        return {
            "text": "",
            "quality_gain": 0.0,
            "faithfulness": 0.0,
            "notes": str(exc),
        }

    path = str(payload.get("path") or "")
    if not path:
        formatted = json.dumps(source, indent=2, ensure_ascii=False)
        return {
            "text": formatted,
            "quality_gain": 0.05,
            "faithfulness": 1.0,
            "notes": "returned whole json",
        }

    try:
        tokens = _tokenize_path(path)
        value = _traverse_path(source, tokens)
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        return {
            "text": "",
            "quality_gain": 0.0,
            "faithfulness": 0.0,
            "notes": f"path error: {exc}",
        }
    formatted = json.dumps(value, indent=2, ensure_ascii=False)
    return {
        "text": formatted,
        "quality_gain": 0.06,
        "faithfulness": 1.0,
        "notes": f"path '{path}' resolved",
    }


def dedupe_lines(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = str(payload.get("text") or "")
    seen = set()
    output_lines: List[str] = []
    for line in text.splitlines():
        if line not in seen:
            seen.add(line)
            output_lines.append(line)
    return {
        "text": "\n".join(output_lines),
        "quality_gain": 0.03 if output_lines else 0.0,
        "faithfulness": 1.0,
        "notes": f"{len(output_lines)} unique line(s)",
    }


__all__ = [
    "table_detect",
    "csv_summary",
    "json_query",
    "dedupe_lines",
]
