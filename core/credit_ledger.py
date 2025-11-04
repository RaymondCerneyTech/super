from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

LEDGER_PATH = Path(".ai") / "ledger_tool_calls.jsonl"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def record(call_site: str, tool: str, score: Dict[str, Any], meta: Dict[str, Any]) -> None:
    payload: Dict[str, Any] = {
        "ts": _timestamp(),
        "call_site": call_site,
        "tool": tool,
        "ok": bool(score.get("ok")),
        "delta_quality": float(score.get("delta_quality", 0.0)),
        "latency_ms": float(meta.get("latency_ms", 0.0)),
        "faithfulness": float(score.get("faithfulness", 0.0)),
        "notes": str(score.get("notes", "")),
    }
    payload.update({key: meta[key] for key in ("episode", "cue") if key in meta})

    path = LEDGER_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.write("\n")
    except OSError:
        pass


__all__ = ["record"]
