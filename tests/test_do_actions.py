from __future__ import annotations

import os
from pathlib import Path

import pytest

from main import command_do


class Args:
    def __init__(self, **kwargs):
        self.task = kwargs.get("task")
        self.workspace = kwargs.get("workspace")
        self.approve = kwargs.get("approve", False)
        self.permit = kwargs.get("permit", "")


def test_do_dry_run_preview(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    workspace = tmp_path / "ws"
    args = Args(task="Summarize docs/input.txt to docs/output.txt", workspace=str(workspace))
    command_do(args)
    out = capsys.readouterr().out
    assert "Plan preview" in out
    assert "Dry run only" in out
    assert not (workspace / "docs" / "output.txt").exists()


def test_do_execute_summary(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    input_path = workspace / "docs" / "input.txt"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text("Artificial intelligence enables automation and better decisions.\n" * 2, encoding="utf-8")

    args = Args(
        task="Summarize docs/input.txt to docs/output.txt",
        workspace=str(workspace),
        approve=True,
        permit="read,write",
    )
    command_do(args)
    output_path = workspace / "docs" / "output.txt"
    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8").strip()
