from __future__ import annotations

from typing import Iterable, Optional, Set

from core.interfaces import Behavior, Context, Result
from core.sandbox import Permission, create_sandbox


class HttpDownload(Behavior):
    name = "http_download"
    inputs = ["url", "dst"]
    outputs = ["download_info"]

    def run(self, ctx: Context) -> Result:
        data = ctx.setdefault("data", {})
        perms = _parse_permissions(ctx.get("perms"))
        dry_run = bool(ctx.get("dry_run", True))
        url = data.get("url") or ctx.get("url")
        dst = data.get("dst") or ctx.get("dst")
        expected_hash = data.get("sha256") or ctx.get("sha256")

        if not url or not dst:
            return _error_result("Both url and dst are required")

        sandbox = create_sandbox(ctx.get("workspace", "workspace"), permissions=perms)
        if Permission.NET not in sandbox.permissions:
            return _error_result("Missing net permission")
        if Permission.WRITE not in sandbox.permissions:
            return _error_result("Missing write permission")

        if dry_run:
            log = f"[dry-run] would download {url} -> {dst}"
            return _success(log, dst, None, None)

        result = sandbox.net_download(url, dst)
        sha256 = result.get("sha256")
        if expected_hash and sha256 and sha256.lower() != str(expected_hash).lower():
            return _error_result("Downloaded file hash mismatch")

        log = f"Downloaded {url} -> {result['path']}"
        return _success(log, str(result.get("path")), sha256, result.get("size"))


__all__ = ["HttpDownload"]


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


def _success(log: str, path: Optional[str], sha256: Optional[str], size: Optional[int]) -> Result:
    return {
        "ok": True,
        "logs": [log],
        "reward": 1.0,
        "effects": ["downloaded"],
        "rationale": {"why": log, "evidence": [e for e in [path, sha256] if e]},
        "output": {
            "download_info": {
                "path": path,
                "sha256": sha256,
                "size": size,
            }
        },
    }


def _error_result(message: str) -> Result:
    return {
        "ok": False,
        "logs": [message],
        "reward": 0.0,
        "effects": [],
        "rationale": {"why": message, "evidence": []},
    }
