from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REFLECTIONS_PATH = Path(".ai") / "reflections.jsonl"


def fingerprint(text: str) -> str:
    normalized = " ".join(str(text or "").lower().split())
    if not normalized:
        return "empty"
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def _load() -> Dict[Tuple[str, str, str], Dict[str, object]]:
    records: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    if not REFLECTIONS_PATH.exists():
        return records
    try:
        for line in REFLECTIONS_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            key = (payload.get("cue", ""), payload.get("fingerprint", ""), payload.get("reflection", ""))
            records[key] = payload
    except (OSError, json.JSONDecodeError):
        pass
    return records


def _write(records: Iterable[Dict[str, object]]) -> None:
    REFLECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REFLECTIONS_PATH.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record, handle)
            handle.write("\n")


def add_reflection(cue: str, fp: str, note: str, ok: bool) -> None:
    store = _load()
    key = (cue, fp, note)
    entry = store.get(
        key,
        {
            "cue": cue,
            "fingerprint": fp,
            "reflection": note,
            "win": 0,
            "total": 0,
        },
    )
    entry["total"] = int(entry.get("total", 0)) + 1
    if ok:
        entry["win"] = int(entry.get("win", 0)) + 1
    store[key] = entry
    _write(store.values())


def get_reflections(cue: str, fp: str, limit: int = 3) -> List[str]:
    store = _load()
    candidates = [payload for payload in store.values() if payload.get("cue") == cue and payload.get("fingerprint") == fp]
    def _score(rec: Dict[str, object]) -> Tuple[float, int]:
        win = int(rec.get("win", 0) or 0)
        total = int(rec.get("total", 0) or 0)
        ratio = (win + 1) / (total + 1)
        return (ratio, total)

    candidates.sort(key=_score, reverse=True)
    return [str(entry.get("reflection", "")) for entry in candidates[:limit]]


__all__ = ["fingerprint", "add_reflection", "get_reflections", "REFLECTIONS_PATH"]
