from __future__ import annotations

from behaviors.command_parse import CommandParse
from core.interfaces import Context


def run_task(task: str):
    ctx: Context = {
        "text": task,
        "data": {"task": task}
    }
    return CommandParse().run(ctx)


def test_parse_download():
    result = run_task("Download https://example.com/file.zip to downloads/file.zip")
    steps = result["output"]["steps"]
    assert steps and steps[0]["behavior"] == "http_download"
    assert steps[0]["args"]["dst"] == "downloads/file.zip"


def test_parse_zip_and_unzip():
    task = "Zip src/folder to archives/folder.zip and unzip archives/folder.zip to output/"
    result = run_task(task)
    steps = result["output"]["steps"]
    assert any(step["behavior"] == "zip_ops" and step["args"]["op"] == "zip" for step in steps)
    assert any(step["behavior"] == "zip_ops" and step["args"]["op"] == "unzip" for step in steps)


def test_parse_summarize_sequence():
    result = run_task("Summarize docs/input.txt to docs/summary.txt")
    steps = result["output"]["steps"]
    behaviors = [step["behavior"] for step in steps]
    assert behaviors == ["files_read", "summarize", "files_write"]


def test_parse_list_files():
    result = run_task("List files under docs/ with *.pdf")
    steps = result["output"]["steps"]
    assert steps[0]["behavior"] == "files_list"
    assert steps[0]["args"]["glob"] == "*.pdf"
