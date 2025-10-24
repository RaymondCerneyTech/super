from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class PlanBlueprint:
    behaviors: List[str]
    created_ts: float


class PlanCache:
    def __init__(self, path: Optional[Path] = None, ttl_seconds: Optional[int] = None) -> None:
        self.path = path or self._default_path()
        self.ttl_seconds = ttl_seconds or self._default_ttl()
        self._cache: Dict[str, PlanBlueprint] = {}
        self._load()

    def _default_path(self) -> Path:
        env_path = os.getenv("SUPER_PLAN_CACHE_PATH")
        if env_path:
            return Path(env_path)
        env_dir = os.getenv("SUPER_PLAN_CACHE_DIR")
        base_dir = Path(env_dir) if env_dir else Path("data") / "plans"
        return base_dir / "blueprints.json"

    def _default_ttl(self) -> int:
        override = os.getenv("SUPER_PLAN_CACHE_TTL")
        if override:
            try:
                value = int(override)
                if value >= 0:
                    return value
            except ValueError:
                pass
        return 3600  # 1 hour default

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return
        now = time.time()
        for key, entry in payload.items():
            behaviors = entry.get("behaviors")
            created_ts = entry.get("created_ts")
            if not isinstance(behaviors, list) or not isinstance(created_ts, (int, float)):
                continue
            if self.ttl_seconds and created_ts + self.ttl_seconds < now:
                continue
            self._cache[key] = PlanBlueprint(list(map(str, behaviors)), float(created_ts))

    def _persist(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                key: {"behaviors": blueprint.behaviors, "created_ts": blueprint.created_ts}
                for key, blueprint in self._cache.items()
            }
            with self.path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle)
        except OSError:
            return

    def build_key(
        self,
        goal_flags: Iterable[str],
        *,
        backend: str = "tfidf",
        verbosity: str = "normal",
        tags: Optional[Iterable[str]] = None,
    ) -> str:
        flags = ",".join(sorted({flag.lower() for flag in goal_flags}))
        tags_key = ",".join(sorted({str(tag).strip().lower() for tag in (tags or []) if str(tag).strip()}))
        fingerprint = f"flags={flags}|backend={backend.lower()}|verbosity={verbosity.lower()}|tags={tags_key}"
        return fingerprint

    def lookup(self, key: str) -> Optional[List[str]]:
        entry = self._cache.get(key)
        if not entry:
            return None
        if self.ttl_seconds and entry.created_ts + self.ttl_seconds < time.time():
            return None
        return list(entry.behaviors)

    def store(self, key: str, behaviors: List[Tuple[str, object]]) -> None:
        behavior_names = [name for name, _ in behaviors if isinstance(name, str)]
        if not behavior_names:
            return
        self._cache[key] = PlanBlueprint(behaviors=behavior_names, created_ts=time.time())
        self._persist()

    def clear(self) -> None:
        self._cache.clear()
        if self.path.exists():
            try:
                self.path.unlink()
            except OSError:
                pass


_SHARED_CACHE: Optional[PlanCache] = None


def shared_plan_cache() -> PlanCache:
    global _SHARED_CACHE
    if _SHARED_CACHE is None:
        _SHARED_CACHE = PlanCache()
    return _SHARED_CACHE


__all__ = ["PlanCache", "shared_plan_cache"]