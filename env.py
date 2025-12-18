from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple


@dataclass
class Task:
    name: str
    initial_solution: str
    tests: str
    max_steps: int = 8


@dataclass
class CodeWorld:
    task: Task
    temp_dir: Path = field(init=False)
    step_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix=f"{self.task.name}_"))
        self._write_file("solution.py", self.task.initial_solution)
        self._write_file("test_solution.py", self.task.tests)

    def summarize_state(self) -> str:
        head = self._read_file("solution.py")[:400]
        summary = {
            "task_name": self.task.name,
            "step": self.step_count,
            "max_steps": self.task.max_steps,
            "solution_head": head,
        }
        return json.dumps(summary)

    def step(self, new_solution_code: str) -> Tuple[Dict[str, object], float, bool, Dict[str, object]]:
        self._write_file("solution.py", new_solution_code)
        self.step_count += 1
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=self.temp_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        raw_output = result.stdout + result.stderr
        passed, failed, total = self._parse_pytest_output(result.stdout)
        reward = passed / max(total, 1)
        done = (failed == 0 and passed > 0) or self.step_count >= self.task.max_steps
        truncated = raw_output[:1000]
        obs = {
            "passed": passed,
            "failed": failed,
            "total": total,
            "test_output": truncated,
        }
        info = {"raw_output": raw_output}
        return obs, reward, done, info

    def cleanup(self) -> None:
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def _write_file(self, filename: str, contents: str) -> None:
        (self.temp_dir / filename).write_text(contents, encoding="utf-8")

    def _read_file(self, filename: str) -> str:
        return (self.temp_dir / filename).read_text(encoding="utf-8")

    @staticmethod
    def _parse_pytest_output(output: str) -> Tuple[int, int, int]:
        passed = failed = total = 0
        markers = [token for token in output.replace("=", " ").split() if token.isdigit()]
        if markers:
            total = int(markers[-1])
        summary_line = next((line for line in output.splitlines() if "passed" in line or "failed" in line), "")
        parts = summary_line.replace(",", "").split()
        for idx, token in enumerate(parts):
            if token == "passed":
                try:
                    passed = int(parts[idx - 1])
                except (IndexError, ValueError):
                    continue
            elif token == "failed":
                try:
                    failed = int(parts[idx - 1])
                except (IndexError, ValueError):
                    continue
        total = max(total, passed + failed)
        return passed, failed, total
