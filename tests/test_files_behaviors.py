from __future__ import annotations

from pathlib import Path

import pytest

from behaviors.files_read import FilesRead
from behaviors.files_write import FilesWrite
from behaviors.files_list import FilesList
from behaviors.files_search import FilesSearch
from behaviors.files_move_copy_delete import FilesMoveCopyDelete
from behaviors.zip_ops import ZipOps
from behaviors.http_download import HttpDownload


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    return tmp_path


def test_files_write_and_read(workspace: Path) -> None:
    write_ctx = {
        "workspace": workspace,
        "perms": ["write"],
        "dry_run": False,
        "data": {"path": "docs/hello.txt", "content": "hello"},
    }
    write_result = FilesWrite().run(write_ctx)
    assert write_result["ok"]
    target = workspace / "docs" / "hello.txt"
    assert target.exists()
    read_ctx = {
        "workspace": workspace,
        "perms": ["read"],
        "dry_run": False,
        "data": {"path": "docs/hello.txt"},
    }
    read_result = FilesRead().run(read_ctx)
    assert read_result["ok"]
    assert read_result["output"]["text"] == "hello"


def test_files_list_and_search(workspace: Path) -> None:
    (workspace / "docs").mkdir(parents=True)
    (workspace / "docs" / "fileA.txt").write_text("A")
    (workspace / "docs" / "fileB.log").write_text("B")

    list_ctx = {
        "workspace": workspace,
        "perms": ["read"],
        "dry_run": False,
        "data": {"path": "docs", "glob": "*.txt"},
    }
    list_result = FilesList().run(list_ctx)
    paths = list_result["output"]["paths"]
    assert any(path.endswith("fileA.txt") for path in paths)

    search_ctx = {
        "workspace": workspace,
        "perms": ["read"],
        "dry_run": False,
        "data": {"path": "docs", "pattern": "fileB"},
    }
    search_result = FilesSearch().run(search_ctx)
    matches = search_result["output"]["matches"]
    assert any(match.endswith("fileB.log") for match in matches)


def test_move_copy_delete(workspace: Path) -> None:
    src = workspace / "docs"
    src.mkdir()
    (src / "original.txt").write_text("content")

    copy_ctx = {
        "workspace": workspace,
        "perms": ["read", "write"],
        "dry_run": False,
        "data": {"op": "copy", "src": "docs/original.txt", "dst": "docs/copy.txt"},
    }
    FilesMoveCopyDelete().run(copy_ctx)
    assert (workspace / "docs" / "copy.txt").exists()

    move_ctx = {
        "workspace": workspace,
        "perms": ["write"],
        "dry_run": False,
        "data": {"op": "move", "src": "docs/copy.txt", "dst": "docs/moved.txt"},
    }
    FilesMoveCopyDelete().run(move_ctx)
    assert not (workspace / "docs" / "copy.txt").exists()
    assert (workspace / "docs" / "moved.txt").exists()

    delete_ctx = {
        "workspace": workspace,
        "perms": ["write"],
        "dry_run": False,
        "data": {"op": "delete", "src": "docs/moved.txt"},
    }
    FilesMoveCopyDelete().run(delete_ctx)
    assert not (workspace / "docs" / "moved.txt").exists()


def test_zip_ops(workspace: Path) -> None:
    src_dir = workspace / "folder"
    src_dir.mkdir()
    (src_dir / "a.txt").write_text("data")

    zip_ctx = {
        "workspace": workspace,
        "perms": ["read", "write"],
        "dry_run": False,
        "data": {"op": "zip", "src": "folder", "dst": "archive.zip"},
    }
    zip_result = ZipOps().run(zip_ctx)
    archive_path = Path(zip_result["output"]["path"])
    assert archive_path.exists()

    unzip_ctx = {
        "workspace": workspace,
        "perms": ["read", "write"],
        "dry_run": False,
        "data": {"op": "unzip", "src": "archive.zip", "dst": "extracted"},
    }
    ZipOps().run(unzip_ctx)
    assert (workspace / "extracted" / "a.txt").exists()


def test_http_download_dry_run(workspace: Path) -> None:
    ctx = {
        "workspace": workspace,
        "perms": ["net", "write"],
        "dry_run": True,
        "data": {"url": "https://example.com/file.txt", "dst": "downloads/file.txt"},
    }
    result = HttpDownload().run(ctx)
    assert result["ok"]
    assert not (workspace / "downloads" / "file.txt").exists()
