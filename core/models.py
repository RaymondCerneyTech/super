from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

MODEL_EXTENSIONS: Sequence[str] = (
    ".gguf",
    ".bin",
    ".pt",
    ".pth",
    ".safetensors",
    ".onnx",
    ".json",
    ".yaml",
    ".yml",
)

STATE_PATH = Path(".ai") / "models_state.json"


def model_root() -> Path:
    env = os.getenv("SUPER_MODELS_ROOT")
    if env:
        return Path(env)
    return Path("models")


def list_models() -> List[Path]:
    root = model_root()
    if not root.exists():
        return []
    entries: List[Path] = []
    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if child.is_dir():
            entries.append(child)
        elif child.suffix.lower() in MODEL_EXTENSIONS:
            entries.append(child)
    return entries


def _load_state() -> Dict[str, str]:
    if not STATE_PATH.exists():
        return {}
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_state(state: Dict[str, str]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATE_PATH.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)


def get_active_model() -> Optional[Path]:
    state = _load_state()
    active = state.get("active_model")
    if not active:
        return None
    path = Path(active)
    if not path.exists():
        return None
    return path


def set_active_model(target: Path) -> Path:
    resolved = target.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Model not found: {resolved}")
    _save_state({"active_model": str(resolved)})
    return resolved


def find_model(name: str) -> Optional[Path]:
    candidates = list_models()
    for candidate in candidates:
        if candidate.name == name:
            return candidate
    # allow partial matches (prefix)
    matches = [c for c in candidates if name.lower() in c.name.lower()]
    if len(matches) == 1:
        return matches[0]
    return None


def select_model_by_index(index: int) -> Path:
    models = list_models()
    if not models:
        raise FileNotFoundError("No models found in SUPER_MODELS_ROOT")
    if index < 1 or index > len(models):
        raise IndexError(f"Model index {index} out of range (1..{len(models)})")
    return set_active_model(models[index - 1])


def select_model_by_name(name: str) -> Path:
    match = find_model(name)
    if not match:
        raise FileNotFoundError(f"Model '{name}' not found under {model_root()}")
    return set_active_model(match)


def resolve_model_path(identifier: Optional[str]) -> Path:
    if identifier:
        candidate = Path(identifier)
        if candidate.exists():
            return candidate.resolve()
        match = find_model(identifier)
        if match:
            return match.resolve()
        raise FileNotFoundError(f"Model '{identifier}' not found. Use `main.py models --list` to view available models.")
    active = get_active_model()
    if active:
        return active.resolve()
    raise FileNotFoundError(
        "No active model configured. Use `main.py models --select ...` or pass --model with an explicit path."
    )


__all__ = [
    "model_root",
    "list_models",
    "get_active_model",
    "set_active_model",
    "select_model_by_index",
    "select_model_by_name",
    "resolve_model_path",
]
