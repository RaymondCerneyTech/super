from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

TRACE_PATH = Path(".ai") / "srl_traces.jsonl"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_trace(entry: Dict[str, Any]) -> None:
    payload = dict(entry)
    payload.setdefault("ts", _timestamp())
    try:
        TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with TRACE_PATH.open("a", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.write("\n")
    except OSError:
        pass


__all__ = ["append_trace"]

