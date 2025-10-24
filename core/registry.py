from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from core.interfaces import Behavior


class BehaviorRegistry:
    """Registry that discovers behavior classes and their metadata."""

    DEFAULT_META: Dict[str, object] = {
        "preconditions": ["true"],
        "effects": [],
        "cost": 1.0,
        "capabilities": ["analytic"],
        "requires_explanation": False,
        "success_checks": [],
        "keywords": [],
    }

    def __init__(self) -> None:
        self._behaviors: Dict[str, Behavior] = {}
        self._meta: Dict[str, Dict] = {}
        self._package_dir: Optional[Path] = None

    def discover(self, package_dir: str = "behaviors") -> "BehaviorRegistry":
        """Locate behavior modules under `package_dir` and instantiate them."""
        base_path = Path(package_dir)
        if not base_path.is_absolute():
            project_root = Path(__file__).resolve().parent.parent
            base_path = project_root / base_path

        if not base_path.exists():
            return self

        self._package_dir = base_path
        self._behaviors.clear()

        for module_path in sorted(base_path.glob("*.py")):
            if module_path.name == "__init__.py":
                continue

            module = self._import_module_from_path(module_path)
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if not issubclass(obj, Behavior) or obj is Behavior:
                    continue

                instance = obj()  # type: ignore[call-arg]
                behavior_name = getattr(instance, "name", None) or obj.__name__
                self._behaviors[behavior_name] = instance

        return self

    def load_meta(self) -> "BehaviorRegistry":
        """Load `.meta.yaml` files that share the behavior basename."""
        if self._package_dir is None:
            self.discover()

        if self._package_dir is None:
            return self

        self._meta.clear()
        for name in self._behaviors:
            meta_path = self._package_dir / f"{name}.meta.yaml"
            if meta_path.exists():
                with meta_path.open("r", encoding="utf-8") as fh:
                    raw = yaml.safe_load(fh) or {}
            else:
                raw = {}
            self._meta[name] = self._normalize_meta(raw)

        return self

    def get(self, name: str) -> Behavior:
        return self._behaviors[name]

    def meta(self, name: str) -> Dict:
        return self._meta.get(name, dict(self.DEFAULT_META))

    def list(self) -> List[str]:
        return sorted(self._behaviors.keys())

    def behavior_capabilities(self, name: str) -> List[str]:
        meta = self.meta(name)
        return list(meta.get("capabilities", ["analytic"]))

    def behavior_cost(self, name: str) -> float:
        meta = self.meta(name)
        try:
            return float(meta.get("cost", 1.0))
        except (TypeError, ValueError):
            return 1.0

    def behavior_effects(self, name: str) -> List[str]:
        meta = self.meta(name)
        effects = meta.get("effects", [])
        return list(effects) if isinstance(effects, list) else []

    def behavior_preconditions(self, name: str) -> List[str]:
        meta = self.meta(name)
        preconds = meta.get("preconditions", ["true"])
        return list(preconds) if isinstance(preconds, list) else ["true"]

    def behavior_requires_explanation(self, name: str) -> bool:
        meta = self.meta(name)
        return bool(meta.get("requires_explanation", False))

    def behavior_cluster(self, name: str) -> str:
        caps = [cap.lower() for cap in self.behavior_capabilities(name)]
        return "creative" if "creative" in caps else "analytic"

    @staticmethod
    def _import_module_from_path(module_path: Path):
        module_name = f"{module_path.parent.name}.{module_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        return module

    def _normalize_meta(self, raw: Dict[str, object]) -> Dict[str, object]:
        normalized = dict(self.DEFAULT_META)
        normalized.update(raw or {})

        for key in ("preconditions", "effects", "capabilities", "keywords", "success_checks"):
            value = normalized.get(key)
            if not isinstance(value, list):
                normalized[key] = [value] if value not in (None, "") else []

        try:
            normalized["cost"] = float(normalized.get("cost", 1.0))
        except (TypeError, ValueError):
            normalized["cost"] = 1.0

        normalized["requires_explanation"] = bool(normalized.get("requires_explanation", False))
        return normalized  # type: ignore[return-value]


__all__ = ["BehaviorRegistry"]
