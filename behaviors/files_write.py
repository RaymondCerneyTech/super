from __future__ import annotations

from typing import Iterable, Set

from core.interfaces import Behavior, Context, Result
from core.sandbox import Permission, create_sandbox


class FilesWrite(Behavior):
    name = "files_write"
    inputs = ["path"]
    outputs: list[str] = []

    def run(self, ctx: Context) -> Result:
        data = ctx.setdefault("data", {})
        perms = _parse_permissions(ctx.get("perms"))
        dry_run = bool(ctx.get("dry_run", True))
        path = data.get("path") or ctx.get("path")
        content_key = data.get("content_key") or ctx.get("content_key")
        content = data.get("content") or ctx.get("content")
        if content is None and content_key:
            content = data.get(content_key)
        content = str(content or "")
        overwrite = _get_flag(ctx, data, "overwrite", False)
        create_dirs = _get_flag(ctx, data, "create_dirs", True)

        if not path:
            return _error_result("No path provided")
        if Permission.WRITE not in perms:
            return _error_result("Missing write permission")

        if dry_run:
            return {
                "ok": True,
                "logs": [f"[dry-run] would write {path}"],
                "reward": 1.0,
                "effects": ["file_written"],
                "rationale": {"why": "dry run", "evidence": [path]},
                "output": {"path": path, "bytes": len(str(content).encode("utf-8"))},
            }

        sandbox = create_sandbox(ctx.get("workspace", "workspace"), permissions=perms)
        target = sandbox.fs_write(path, content, create_dirs=create_dirs, overwrite=overwrite)
        return {
            "ok": True,
            "logs": [f"Wrote {target}"],
            "reward": 1.0,
            "effects": ["file_written"],
            "rationale": {"why": "wrote file", "evidence": [str(target)]},
            "output": {"path": str(target), "bytes": len(str(content).encode("utf-8"))},
        }


__all__ = ["FilesWrite"]


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


def _get_flag(ctx: Context, data: dict, key: str, default: bool) -> bool:
    if key in data and data[key] is not None:
        return bool(data[key])
    if key in ctx and ctx[key] is not None:
        return bool(ctx[key])
    return default


def _error_result(message: str) -> Result:
    return {
        "ok": False,
        "logs": [message],
        "reward": 0.0,
        "effects": [],
        "rationale": {"why": message, "evidence": []},
    }
