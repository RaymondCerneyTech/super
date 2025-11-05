from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional

DEFAULT_BIN_NAMES = [
    "llama",
    "llama.exe",
    "main",
    "main.exe",
    "llama-cli",
    "llama-cli.exe",
]


class LlamaBinaryNotFound(FileNotFoundError):
    """Raised when the llama executable cannot be located."""


def _candidate_directories() -> List[Path]:
    dirs: List[Path] = []
    env_root = os.getenv("LLAMA_ROOT")
    if env_root:
        root = Path(env_root)
        dirs.append(root)
        dirs.append(root / "bin")
        dirs.append(root / "build")
    # Allow override
    env_bin = os.getenv("LLAMA_BIN")
    if env_bin:
        dirs.append(Path(env_bin).parent)
    # Local project fallbacks
    dirs.append(Path("llama"))
    dirs.append(Path("llama") / "build")
    dirs.append(Path.cwd())
    return dirs


def find_llama_binary() -> Path:
    env_override = os.getenv("LLAMA_BIN")
    if env_override:
        candidate = Path(env_override)
        if candidate.exists():
            return candidate.resolve()
    for directory in _candidate_directories():
        if not directory.exists():
            continue
        for name in DEFAULT_BIN_NAMES:
            candidate = directory / name
            if candidate.exists():
                return candidate.resolve()
    raise LlamaBinaryNotFound(
        "Unable to locate llama binary. Set LLAMA_ROOT or LLAMA_BIN to point to your llama.cpp build."
    )


def run_inference(
    *,
    prompt: str,
    model: Path,
    n_predict: Optional[int] = None,
    temperature: Optional[float] = None,
    extra_args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    binary = find_llama_binary()
    args = [str(binary), "--model", str(model)]
    if n_predict is not None:
        args.extend(["--n-predict", str(n_predict)])
    if temperature is not None:
        args.extend(["--temp", str(temperature)])
    args.extend(["--prompt", prompt])
    if extra_args:
        args.extend(extra_args)

    proc_env = None
    if env:
        proc_env = os.environ.copy()
        proc_env.update(env)

    completed = subprocess.run(
        args,
        capture_output=True,
        text=True,
        env=proc_env,
        encoding="utf-8",
        errors="replace",
    )
    return {
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "returncode": str(completed.returncode),
        "command": " ".join(shlex.quote(a) for a in args),
    }


__all__ = ["find_llama_binary", "run_inference", "LlamaBinaryNotFound"]
