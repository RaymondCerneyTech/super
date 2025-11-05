from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml

DEFAULT_PROFILES: Dict[str, object] = {
    "default_model": "",
    "profiles": {
        "poem": {
            "description": "Generate a short two-line poem.",
            "prompt": "Write a two line poem about {topic}.",
            "n_predict": 60,
            "temperature": 0.7,
            "extra": [
                "--simple-io",
                "--no-warmup",
                "-no-cnv",
                "--top-k=40",
                "--top-p=0.9",
                "--repeat-penalty=1.1",
            ],
        }
    },
}

CONFIG_PATHS: Tuple[Path, ...] = (Path("config") / "llama_profiles.yaml", Path("llama_profiles.yaml"))
CONFIG_ENV = "SUPER_LLAMA_PROFILES"


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _merge(dest: Dict[str, object], src: Dict[str, object]) -> None:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dest.get(key), dict):
            _merge(dest[key], value)  # type: ignore[arg-type]
        else:
            dest[key] = value


def _load_yaml(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("llama profile config must be a mapping")
    return payload  # type: ignore[return-value]


def load_profiles() -> Dict[str, object]:
    data = copy.deepcopy(DEFAULT_PROFILES)
    paths: List[Path] = []
    env_path = os.getenv(CONFIG_ENV)
    if env_path:
        paths.append(Path(env_path))
    paths.extend(CONFIG_PATHS)
    for path in paths:
        if path.exists():
            try:
                payload = _load_yaml(path)
            except Exception:
                continue
            _merge(data, payload)
    return data


def list_profiles() -> List[Tuple[str, str]]:
    data = load_profiles()
    profiles = data.get("profiles", {}) if isinstance(data, dict) else {}
    output: List[Tuple[str, str]] = []
    if isinstance(profiles, dict):
        for name, payload in profiles.items():
            description = ""
            if isinstance(payload, dict):
                description = str(payload.get("description", "") or "")
            output.append((str(name), description))
    return output


def resolve_profile(
    name: str,
    variables: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    data = load_profiles()
    profiles = data.get("profiles", {}) if isinstance(data, dict) else {}
    if not isinstance(profiles, dict) or name not in profiles:
        raise KeyError(f"Unknown llama profile '{name}'")
    profile = copy.deepcopy(profiles[name]) if isinstance(profiles[name], dict) else {}
    defaults_model = data.get("default_model") if isinstance(data, dict) else ""
    if isinstance(profile, dict):
        rendered_prompt = profile.get("prompt")
        if isinstance(rendered_prompt, str):
            vars_safe = _SafeDict(variables or {})
            profile["prompt"] = rendered_prompt.format_map(vars_safe)
        if "model" not in profile and isinstance(defaults_model, str) and defaults_model:
            profile["model"] = defaults_model
    if not isinstance(profile, dict):
        profile = {}
    return profile


def render_template(template: str, variables: Optional[Dict[str, str]] = None) -> str:
    return template.format_map(_SafeDict(variables or {}))


__all__ = ["load_profiles", "list_profiles", "resolve_profile", "render_template"]
