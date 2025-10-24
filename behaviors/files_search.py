from __future__ import annotations

from typing import Iterable, Set

from core.interfaces import Behavior, Context, Result
from core.sandbox import Permission, create_sandbox


class FilesSearch(Behavior):
    name = "files_search"
    inputs = ["path", "pattern"]
    outputs = ["matches"]

    def run(self, ctx: Context) -> Result:
        data = ctx.setdefault("data", {})
        perms = _parse_permissions(ctx.get("perms"))
        dry_run = bool(ctx.get("dry_run", True))
        path = data.get("path") or ctx.get("path")
        pattern = (data.get("pattern") or ctx.get("pattern") or "").strip()
        recursive = _get_flag(ctx, data, "recursive", True)

        if not path:
            return _error_result("No path provided")
        if not pattern:
            return _error_result("No pattern provided")
        if Permission.READ not in perms:
            return _error_result("Missing read permission")

        sandbox = create_sandbox(ctx.get("workspace", "workspace"), permissions=perms)
        entries = [str(p) for p in sandbox.fs_list(path, recursive=recursive, glob="*")]
        matches = [p for p in entries if pattern.lower() in p.lower()]

        if dry_run:
            log = f"[dry-run] would search {path} (found {len(matches)} matches)"
        else:
            log = f"Found {len(matches)} matches under {path}"

        return {
            "ok": True,
            "logs": [log],
            "reward": 1.0,
            "effects": ["have_matches"],
            "rationale": {"why": "searched paths", "evidence": [pattern]},
            "output": {"matches": matches},
        }


__all__ = ["FilesSearch"]


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
