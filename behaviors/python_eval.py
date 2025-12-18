from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from core.interfaces import Behavior, Context, Result


class PythonEval(Behavior):
    name = "python_eval"
    inputs: List[str] = []
    outputs: List[str] = ["python_eval"]

    def run(self, ctx: Context) -> Result:
        data = ctx.setdefault("data", {})
        code = data.get("code") or ctx.get("code")
        if not isinstance(code, str) or not code.strip():
            return {
                "ok": False,
                "logs": ["python_eval: missing code snippet"],
                "reward": 0.0,
                "rewards": {"overall": 0.0},
                "effects": [],
            }

        timeout = float(data.get("timeout") or ctx.get("timeout") or 5.0)
        max_output = int(data.get("max_output") or ctx.get("max_output") or 4000)

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
            handle.write(code)
            temp_path = Path(handle.name)

        try:
            completed = subprocess.run(
                [sys.executable, str(temp_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            temp_path.unlink(missing_ok=True)
            return {
                "ok": False,
                "logs": [f"python_eval: execution timed out after {timeout}s"],
                "reward": 0.0,
                "rewards": {"overall": 0.0},
                "effects": [],
            }
        finally:
            temp_path.unlink(missing_ok=True)

        stdout = (completed.stdout or "")[:max_output]
        stderr = (completed.stderr or "")[:max_output]
        outcome = {
            "stdout": stdout.strip(),
            "stderr": stderr.strip(),
            "returncode": completed.returncode,
        }
        data["python_eval"] = outcome

        reward = 1.0 if completed.returncode == 0 else 0.3
        rationale = {
            "why": "Executed Python snippet",
            "evidence": [
                f"returncode={completed.returncode}",
                f"timeout={timeout}",
            ],
        }

        effects = ["python_result"]
        if completed.returncode != 0:
            effects.append("python_error")

        return {
            "ok": True,
            "output": {"python_eval": outcome},
            "logs": ["python_eval executed snippet"],
            "reward": reward,
            "rewards": {"overall": reward},
            "effects": effects,
            "rationale": rationale,
        }


__all__ = ["PythonEval"]

