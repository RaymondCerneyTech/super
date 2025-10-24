from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml

DEFAULT_CONFIG: Dict[str, Any] = {
    "ingest": {
        "index_backend": "tfidf",
        "tags": "",
        "limit": 15,
        "lang_any": False,
    },
    "ask": {
        "index_backend": "tfidf",
        "k_passages": 12,
        "max_chars": 12000,
        "tags": "",
        "verbosity": "verbose",
        "min_words": 800,
        "max_words": None,
        "fresh_days": None,
        "log_file": "logs/bandit.jsonl",
        "max_expansions": 25,
        "explain": False,
        "cited": True,
        "grounded": True,
        "no_bandit": False,
    },
    "plan": {
        "log_file": "logs/bandit.jsonl",
        "max_expansions": 20,
        "explain": False,
        "no_bandit": False,
    },
}

CONFIG_LOCATIONS: Iterable[Path] = (
    Path("super.yaml"),
    Path("config") / "super.yaml",
)


def _merge(dest: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dest.get(key), dict):
            dest[key] = _merge(dest[key], value)
        else:
            dest[key] = value
    return dest


def load_config(path: str | os.PathLike[str] | None = None) -> Dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config_path = _resolve_path(path)
    if not config_path:
        return config
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            user_config = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError):
        return config
    if isinstance(user_config, dict):
        _merge(config, user_config)
    return config


def _resolve_path(path: str | os.PathLike[str] | None) -> Path | None:
    if path:
        candidate = Path(path)
        return candidate if candidate.exists() else None
    env_path = os.getenv("SUPER_CONFIG_PATH")
    if env_path:
        candidate = Path(env_path)
        if candidate.exists():
            return candidate
    for location in CONFIG_LOCATIONS:
        if location.exists():
            return location
    return None


__all__ = ["load_config", "DEFAULT_CONFIG"]
