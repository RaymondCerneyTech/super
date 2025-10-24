from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

RUN_LOG_PATH = Path("runs.log")
ACTION_LOG_PATH = Path("actions.log")


def _write_event(event: Dict[str, Any], log_path: Path) -> None:
    if os.getenv("NO_AUDIT"):
        return
    try:
        with log_path.open("a", encoding="utf-8") as handle:
            json.dump(event, handle, default=_json_default)
            handle.write("\n")
    except OSError:
        pass


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_run(event: Dict[str, Any], log_path: Path = RUN_LOG_PATH) -> None:
    payload = {"ts": _timestamp(), **event}
    _write_event(payload, log_path)


def log_action(event: Dict[str, Any], log_path: Path = ACTION_LOG_PATH) -> None:
    payload = {"ts": _timestamp(), **event}
    _write_event(payload, log_path)
