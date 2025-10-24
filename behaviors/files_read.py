from __future__ import annotations

from typing import Iterable, Set

from core.interfaces import Behavior, Context, Result
from core.sandbox import Permission, create_sandbox


class FilesRead(Behavior):
    name = "files_read"
    inputs = ["path"]
    outputs = ["text"]

    def run(self, ctx: Context) -> Result:
        data = ctx.setdefault("data", {})
        perms = _parse_permissions(ctx.get("perms"))
        dry_run = bool(ctx.get("dry_run", True))
        path = data.get("path") or ctx.get("path")
        if not path:
            return {
                "ok": False,
                "logs": ["No path provided"],
                "reward": 0.0,
                "effects": [],
                "rationale": {"why": "missing path", "evidence": []},
            }

        sandbox = create_sandbox(ctx.get("workspace", "workspace"), permissions=perms)
        if Permission.READ not in sandbox.permissions:
            return {
                "ok": False,
                "logs": ["Missing read permission"],
                "reward": 0.0,
                "effects": [],
                "rationale": {"why": "missing read permission", "evidence": []},
            }

        if dry_run:
            return {
                "ok": True,
                "logs": [f"[dry-run] would read {path}"],
                "reward": 1.0,
                "effects": ["has_text"],
                "rationale": {"why": "dry run", "evidence": [path]},
                "output": {"text": f"[dry-run] preview disabled for {path}"},
            }

        content = sandbox.fs_read(path)
        return {
            "ok": True,
            "logs": [f"Read {path}"],
            "reward": 1.0,
            "effects": ["has_text"],
            "rationale": {"why": "read file", "evidence": [path]},
            "output": {"text": content},
        }


__all__ = ["FilesRead"]


def _parse_permissions(values: Iterable | None) -> Set[Permission]:
    perms: Set[Permission] = set()
    if not values:
        return perms
    for value in values:
        if isinstance(value, Permission):
            perms.add(value)
        elif isinstance(value, str):
            try:
                perms.add(Permission(value))
            except ValueError:
                raise ValueError(f"Unknown permission: {value}") from None
        else:
            raise ValueError(f"Unsupported permission type: {type(value)}")
    return perms
