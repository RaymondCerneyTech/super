from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

LOG_PATH = Path("runs.log")


def log_run(event: Dict[str, Any], log_path: Path = LOG_PATH) -> None:
    if os.getenv("NO_AUDIT"):
        return

    try:
        with log_path.open("a", encoding="utf-8") as handle:
            json.dump(event, handle)
            handle.write("\n")
    except OSError:
        # Silently ignore logging issues to avoid breaking execution.
        pass
