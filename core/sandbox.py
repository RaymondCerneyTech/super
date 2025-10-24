from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Set

import requests

from .audit import log_action


class Permission(str, Enum):
    READ = "read"
    WRITE = "write"
    NET = "net"
    EXEC = "exec"


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    elapsed_ms: int


class SandboxError(Exception):
    """Base class for sandbox related errors."""


class PermissionError(SandboxError):
    """Raised when an operation requires a missing permission."""


class Sandbox:
    COMMAND_ALLOWLIST: Set[str] = {"zip", "unzip", "tar", "python", "node"}

    def __init__(
        self,
        workspace: Path | str = Path("workspace"),
        permissions: Optional[Iterable[Permission]] = None,
        audit_logger: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.permissions: Set[Permission] = set(permissions or [])
        self._audit_logger = audit_logger or log_action

    def _record(self, action: str, **info: Any) -> None:
        if self._audit_logger:
            payload = {"action": action, "workspace": str(self.workspace), **info}
            self._audit_logger(payload)

    @staticmethod
    def _truncate(value: str, limit: int = 1000) -> str:
        if len(value) <= limit:
            return value
        return value[: limit - 3] + "..."

    # ------------------------------------------------------------------ #
    # Permission helpers
    def require(self, *required: Permission) -> None:
        missing = [perm for perm in required if perm not in self.permissions]
        if missing:
            raise PermissionError(f"Missing permissions: {', '.join(perm.value for perm in missing)}")

    # ------------------------------------------------------------------ #
    # Path helpers
    def _resolve_path(self, path: Path | str) -> Path:
        candidate = (self.workspace / Path(path)).resolve() if not Path(path).is_absolute() else Path(path).resolve()
        try:
            candidate.relative_to(self.workspace)
        except ValueError as exc:  # pragma: no cover - defensive
            raise SandboxError(f"Path {candidate} escapes workspace {self.workspace}") from exc
        return candidate

    # ------------------------------------------------------------------ #
    # Command execution
    def run_command(
        self,
        command: Sequence[str],
        *,
        timeout: int = 60,
        env: Optional[dict[str, str]] = None,
    ) -> CommandResult:
        if not command:
            raise SandboxError("No command provided")
        self.require(Permission.EXEC)
        binary = Path(command[0]).name.lower()
        if binary not in self.COMMAND_ALLOWLIST:
            raise SandboxError(f"Command '{binary}' is not in the allowlist")

        start = time.time()
        proc = subprocess.run(
            command,
            cwd=self.workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        elapsed_ms = int((time.time() - start) * 1000)
        result = CommandResult(
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            elapsed_ms=elapsed_ms,
        )
        self._record("command", command=list(command), returncode=result.returncode, elapsed_ms=result.elapsed_ms, stdout=self._truncate(result.stdout), stderr=self._truncate(result.stderr))
        return result

    # ------------------------------------------------------------------ #
    # Filesystem helpers
    def fs_read(self, path: Path | str) -> str:
        self.require(Permission.READ)
        target = self._resolve_path(path)
        content = target.read_text(encoding="utf-8")
        self._record("fs_read", path=str(target))
        return content

    def fs_write(
        self,
        path: Path | str,
        content: str,
        *,
        create_dirs: bool = True,
        overwrite: bool = False,
    ) -> Path:
        self.require(Permission.WRITE)
        target = self._resolve_path(path)
        if target.exists() and not overwrite:
            raise SandboxError(f"File already exists: {target}")
        if create_dirs:
            target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        self._record("fs_write", path=str(target), bytes=len(content.encode("utf-8")), overwrite=overwrite)
        return target

    def fs_list(
        self,
        path: Path | str,
        *,
        recursive: bool = False,
        glob: str | None = None,
    ) -> list[Path]:
        self.require(Permission.READ)
        target = self._resolve_path(path)
        if not target.exists():
            raise SandboxError(f"Path does not exist: {target}")
        pattern = glob or "*"
        results: list[Path] = []
        if target.is_file():
            if target.match(pattern):
                results.append(target)
            return results
        iterator = target.rglob(pattern) if recursive else target.glob(pattern)
        for item in iterator:
            results.append(item)
        self._record("fs_list", path=str(target), recursive=recursive, glob=pattern, count=len(results))
        return results

    def fs_move(self, src: Path | str, dst: Path | str) -> None:
        self.require(Permission.WRITE)
        source = self._resolve_path(src)
        destination = self._resolve_path(dst)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        self._record("fs_move", src=str(source), dst=str(destination))

    def fs_copy(self, src: Path | str, dst: Path | str) -> None:
        self.require(Permission.WRITE)
        source = self._resolve_path(src)
        destination = self._resolve_path(dst)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            if destination.exists():
                raise SandboxError(f"Destination already exists: {destination}")
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
        self._record("fs_copy", src=str(source), dst=str(destination))

    def fs_delete(self, path: Path | str, *, recursive: bool = False) -> None:
        self.require(Permission.WRITE)
        target = self._resolve_path(path)
        if not target.exists():
            return
        if target.is_dir():
            if recursive:
                shutil.rmtree(target)
            else:
                target.rmdir()
        else:
            target.unlink()
        self._record("fs_delete", path=str(target), recursive=recursive)

    def fs_zip(self, src: Path | str, dst_zip: Path | str) -> Path:
        import zipfile

        self.require(Permission.READ, Permission.WRITE)
        source = self._resolve_path(src)
        destination = self._resolve_path(dst_zip).with_suffix(".zip")
        destination.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as zf:
            if source.is_dir():
                for item in source.rglob("*"):
                    if item.is_file():
                        arcname = item.relative_to(source)
                        zf.write(item, arcname)
            else:
                zf.write(source, arcname=source.name)
        self._record("fs_zip", src=str(source), dst=str(destination))
        return destination

    def fs_unzip(self, zip_path: Path | str, dst_dir: Path | str) -> Path:
        import zipfile

        self.require(Permission.WRITE)
        source = self._resolve_path(zip_path)
        destination = self._resolve_path(dst_dir)
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source, "r") as zf:
            zf.extractall(destination)
        self._record("fs_unzip", src=str(source), dst=str(destination))
        return destination

    # ------------------------------------------------------------------ #
    # Network
    def net_download(
        self,
        url: str,
        dst_path: Path | str,
        *,
        timeout: int = 30,
    ) -> dict[str, object]:
        self.require(Permission.NET, Permission.WRITE)
        target = self._resolve_path(dst_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        data = response.content
        target.write_bytes(data)
        sha256 = hashlib.sha256(data).hexdigest()
        result = {"path": target, "sha256": sha256, "size": len(data)}
        self._record("net_download", url=url, dst=str(target), sha256=sha256, size=len(data))
        return result


def create_sandbox(
    workspace: Path | str = Path("workspace"),
    permissions: Optional[Iterable[Permission]] = None,
) -> Sandbox:
    return Sandbox(workspace=workspace, permissions=permissions)
