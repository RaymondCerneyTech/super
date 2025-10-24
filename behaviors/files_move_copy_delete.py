from __future__ import annotations

from typing import Iterable, Set

from core.interfaces import Behavior, Context, Result
from core.sandbox import Permission, create_sandbox


class FilesMoveCopyDelete(Behavior):
    name = "files_move_copy_delete"
    inputs = ["op", "src"]
    outputs: list[str] = []

    def run(self, ctx: Context) -> Result:
        data = ctx.setdefault("data", {})
        perms = _parse_permissions(ctx.get("perms"))
        dry_run = bool(ctx.get("dry_run", True))
        op = (data.get("op") or ctx.get("op") or "").lower()
        src = data.get("src") or ctx.get("src")
        dst = data.get("dst") or ctx.get("dst")
        recursive = _get_flag(ctx, data, "recursive", False)

        if op not in {"move", "copy", "delete"}:
            return _error_result("Unsupported operation")
        if not src:
            return _error_result("Source path required")

        sandbox = create_sandbox(ctx.get("workspace", "workspace"), permissions=perms)

        if op != "delete" and not dst:
            return _error_result("Destination path required")

        if Permission.WRITE not in sandbox.permissions:
            return _error_result("Missing write permission")
        if op == "copy" and Permission.READ not in sandbox.permissions:
            return _error_result("Copy requires read permission")

        if dry_run:
            if op == "delete":
                log = f"[dry-run] would delete {src} (recursive={recursive})"
                evidence = [src]
            else:
                log = f"[dry-run] would {op} {src} -> {dst}"
                evidence = [src, dst]
            return {
                "ok": True,
                "logs": [log],
                "reward": 1.0,
                "effects": ["fs_changed"],
                "rationale": {"why": "dry run", "evidence": evidence},
                "output": {},
            }

        if op == "move":
            sandbox.fs_move(src, dst)
            log = f"Moved {src} -> {dst}"
            evidence = [src, dst]
        elif op == "copy":
            sandbox.fs_copy(src, dst)
            log = f"Copied {src} -> {dst}"
            evidence = [src, dst]
        else:
            sandbox.fs_delete(src, recursive=recursive)
            log = f"Deleted {src}"
            evidence = [src]

        return {
            "ok": True,
            "logs": [log],
            "reward": 1.0,
            "effects": ["fs_changed"],
            "rationale": {"why": log, "evidence": evidence},
            "output": {},
        }


__all__ = ["FilesMoveCopyDelete"]


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
