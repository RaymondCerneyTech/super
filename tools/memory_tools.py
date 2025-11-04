from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict

try:  # Optional dependency on existing core helpers
    from core import reflections as core_reflections  # type: ignore
except Exception:  # pragma: no cover
    core_reflections = None

REFLECTIONS_PATH = Path(".ai") / "reflections.jsonl"
CACHE_DIR = Path(".ai") / "cache"


def reflection_add(payload: Dict[str, Any]) -> Dict[str, Any]:
    cue = str(payload.get("cue") or "")
    fingerprint = str(payload.get("fingerprint") or "")
    note = str(payload.get("note") or "")
    ok = bool(payload.get("ok"))
    if not cue or not fingerprint or not note:
        return {
            "text": "",
            "quality_gain": 0.0,
            "faithfulness": 0.0,
            "notes": "cue, fingerprint, and note required",
        }
    if core_reflections and hasattr(core_reflections, "add_reflection"):
        core_reflections.add_reflection(cue, fingerprint, note, ok)
    else:
        REFLECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": time.time(),
            "cue": cue,
            "fingerprint": fingerprint,
            "reflection": note,
            "ok": ok,
        }
        with REFLECTIONS_PATH.open("a", encoding="utf-8") as handle:
            json.dump(entry, handle)
            handle.write("\n")
    return {
        "text": note,
        "quality_gain": 0.02,
        "faithfulness": 1.0,
        "notes": "reflection stored",
    }


def reflection_get(payload: Dict[str, Any]) -> Dict[str, Any]:
    cue = str(payload.get("cue") or "")
    fingerprint = str(payload.get("fingerprint") or "")
    limit = int(payload.get("k") or payload.get("limit") or 3)
    if not cue or not fingerprint:
        return {
            "text": "",
            "quality_gain": 0.0,
            "faithfulness": 0.0,
            "notes": "cue and fingerprint required",
        }
    hints = []
    if core_reflections and hasattr(core_reflections, "get_reflections"):
        hints = core_reflections.get_reflections(cue, fingerprint, limit=limit)
    elif REFLECTIONS_PATH.exists():
        try:
            with REFLECTIONS_PATH.open("r", encoding="utf-8") as handle:
                records = []
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if entry.get("cue") == cue and entry.get("fingerprint") == fingerprint:
                        records.append(entry)
                records.sort(key=lambda item: item.get("ts", 0), reverse=True)
                hints = [str(item.get("reflection") or item.get("note") or "") for item in records[:limit]]
        except OSError:
            hints = []
    return {
        "text": "\n".join(hints),
        "quality_gain": 0.03 if hints else 0.0,
        "faithfulness": 1.0,
        "notes": f"{len(hints)} reflection(s)",
    }


def _hash_key(key: str) -> str:
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def cache_put(payload: Dict[str, Any]) -> Dict[str, Any]:
    key = payload.get("key")
    if key is None:
        return {
            "text": "",
            "quality_gain": 0.0,
            "faithfulness": 0.0,
            "notes": "key required",
        }
    key_str = str(key)
    value = payload.get("value")
    ttl = payload.get("ttl_seconds")
    record = {
        "value": value,
        "stored_at": time.time(),
        "ttl": float(ttl) if ttl is not None else None,
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / _hash_key(key_str)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle)
    return {
        "text": "OK",
        "quality_gain": 0.02,
        "faithfulness": 1.0,
        "notes": "cached",
    }


def cache_get(payload: Dict[str, Any]) -> Dict[str, Any]:
    key = payload.get("key")
    if key is None:
        return {
            "text": "",
            "quality_gain": 0.0,
            "faithfulness": 0.0,
            "notes": "key required",
        }
    key_str = str(key)
    path = CACHE_DIR / _hash_key(key_str)
    if not path.exists():
        return {
            "text": "",
            "quality_gain": 0.0,
            "faithfulness": 0.0,
            "notes": "cache miss",
        }
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "text": "",
            "quality_gain": 0.0,
            "faithfulness": 0.0,
            "notes": f"cache error: {exc}",
        }
    ttl = record.get("ttl")
    stored_at = record.get("stored_at", 0)
    if ttl is not None and time.time() - stored_at > float(ttl):
        try:
            path.unlink()
        except OSError:
            pass
        return {
            "text": "",
            "quality_gain": 0.0,
            "faithfulness": 0.0,
            "notes": "cache expired",
        }
    value = record.get("value")
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    return {
        "text": text,
        "quality_gain": 0.04,
        "faithfulness": 1.0,
        "notes": "cache hit",
        "value": value,
    }


__all__ = ["reflection_add", "reflection_get", "cache_put", "cache_get"]

