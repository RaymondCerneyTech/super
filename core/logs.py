from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    """Append a JSON object as a single line to the target file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False)
        handle.write("\n")


__all__ = ["append_jsonl"]