from __future__ import annotations

from typing import Iterable, Set

from core.interfaces import Behavior, Context, Result
from core.sandbox import Permission, create_sandbox


class ZipOps(Behavior):
    name = "zip_ops"
    inputs = ["op", "src", "dst"]
    outputs = ["path"]

    def run(self, ctx: Context) -> Result:
        data = ctx.setdefault("data", {})
        perms = _parse_permissions(ctx.get("perms"))
        dry_run = bool(ctx.get("dry_run", True))
        op = (data.get("op") or ctx.get("op") or "").lower()
        src = data.get("src") or ctx.get("src")
        dst = data.get("dst") or ctx.get("dst")

        if op not in {"zip", "unzip"}:
            return _error_result("Unsupported operation")
        if not src or not dst:
            return _error_result("Both src and dst are required")

        sandbox = create_sandbox(ctx.get("workspace", "workspace"), permissions=perms)
        if Permission.READ not in sandbox.permissions or Permission.WRITE not in sandbox.permissions:
            return _error_result("Zip operations require read and write permissions")

        if dry_run:
            log = f"[dry-run] would {op} {src} -> {dst}"
            return {
                "ok": True,
                "logs": [log],
                "reward": 1.0,
                "effects": ["archived" if op == "zip" else "extracted"],
                "rationale": {"why": "dry run", "evidence": [src, dst]},
                "output": {"path": dst},
            }

        if op == "zip":
            target = sandbox.fs_zip(src, dst)
            effect = "archived"
            log = f"Zipped {src} -> {target}"
        else:
            target = sandbox.fs_unzip(src, dst)
            effect = "extracted"
            log = f"Unzipped {src} -> {target}"

        return {
            "ok": True,
            "logs": [log],
            "reward": 1.0,
            "effects": [effect],
            "rationale": {"why": log, "evidence": [src, str(target)]},
            "output": {"path": str(target)},
        }


__all__ = ["ZipOps"]


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


def _error_result(message: str) -> Result:
    return {
        "ok": False,
        "logs": [message],
        "reward": 0.0,
        "effects": [],
        "rationale": {"why": message, "evidence": []},
    }
