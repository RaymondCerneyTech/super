from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Tuple

MAX_TEXT_CHARS = 200_000


def _clean_text(text: object) -> str:
    return text if isinstance(text, str) else str(text or "")


def _read_text_file(path: Path) -> Tuple[str, str]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise IOError(exc) from exc
    if size > MAX_TEXT_CHARS:
        raise IOError("file too large")
    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_TEXT_CHARS + 1)
        if b"\x00" in raw:
            raise IOError("binary file")
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise IOError("failed to decode utf-8") from exc
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]
    return text, f"read {path.name}"


def read_file(payload: Dict[str, object]) -> Dict[str, object]:
    path = Path(str(payload.get("path") or payload.get("text") or ""))
    if not path:
        return {"text": "", "quality_gain": 0.0, "faithfulness": 0.0, "notes": "missing path"}
    try:
        text, note = _read_text_file(path)
        return {"text": text, "quality_gain": 0.1, "faithfulness": 1.0, "notes": note}
    except IOError as exc:
        return {"text": "", "quality_gain": 0.0, "faithfulness": 0.0, "notes": str(exc)}


def write_file(payload: Dict[str, object]) -> Dict[str, object]:
    path = Path(str(payload.get("path") or ""))
    text = _clean_text(payload.get("text") or "")
    if not path:
        return {"text": "", "quality_gain": 0.0, "faithfulness": 0.0, "notes": "missing path"}
    if len(text) > MAX_TEXT_CHARS:
        return {"text": "", "quality_gain": 0.0, "faithfulness": 0.0, "notes": "write aborted: over size limit"}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        return {"text": "", "quality_gain": 0.0, "faithfulness": 0.0, "notes": f"error writing: {exc}"}
    return {"text": "OK", "quality_gain": 0.05, "faithfulness": 1.0, "notes": f"wrote {path.name}"}


def append_file(payload: Dict[str, object]) -> Dict[str, object]:
    path = Path(str(payload.get("path") or ""))
    text = _clean_text(payload.get("text") or "")
    if not path:
        return {"text": "", "quality_gain": 0.0, "faithfulness": 0.0, "notes": "missing path"}
    if len(text) > MAX_TEXT_CHARS:
        return {"text": "", "quality_gain": 0.0, "faithfulness": 0.0, "notes": "append aborted: over size limit"}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text)
    except OSError as exc:
        return {"text": "", "quality_gain": 0.0, "faithfulness": 0.0, "notes": f"error appending: {exc}"}
    return {"text": "OK", "quality_gain": 0.04, "faithfulness": 1.0, "notes": f"appended {len(text)} char(s)"}


def _iter_text_files(root: Path) -> Iterable[Path]:
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            path = Path(dirpath) / filename
            yield path


def grep_repo(payload: Dict[str, object]) -> Dict[str, object]:
    pattern = str(payload.get("pattern") or payload.get("text") or "")
    root = Path(str(payload.get("root") or "."))
    if not pattern:
        return {"text": "", "quality_gain": 0.0, "faithfulness": 0.0, "notes": "empty pattern"}
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        return {"text": "", "quality_gain": 0.0, "faithfulness": 0.0, "notes": f"invalid pattern: {exc}"}

    matches: list[str] = []
    for path in _iter_text_files(root):
        try:
            text, _ = _read_text_file(path)
        except IOError:
            continue
        for idx, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                matches.append(f"{path}:{idx}:{line.strip()}")
    return {
        "text": "\n".join(matches),
        "quality_gain": 0.06 if matches else 0.0,
        "faithfulness": 1.0,
        "notes": f"{len(matches)} match(es)",
    }


def run_pytest(payload: Dict[str, object]) -> Dict[str, object]:
    args = str(payload.get("args") or "").strip()
    import sys

    cmd = [sys.executable, "-m", "pytest"]
    extra_args = shlex.split(args) if args else ["-q"]
    cmd.extend(extra_args)
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        output = proc.stdout.splitlines()
        tail = "\n".join(output[-40:])
        success = proc.returncode == 0
        return {
            "text": tail,
            "quality_gain": 0.2 if success else 0.05,
            "faithfulness": 1.0 if success else 0.7,
            "notes": "pytest completed",
        }
    except OSError as exc:
        return {"text": "", "quality_gain": 0.0, "faithfulness": 0.0, "notes": f"pytest failed: {exc}"}


# Backwards compatibility exports
def grep(payload: Dict[str, object]) -> Dict[str, object]:
    return grep_repo(payload)


__all__ = [
    "read_file",
    "write_file",
    "append_file",
    "grep_repo",
    "grep",
    "run_pytest",
]
