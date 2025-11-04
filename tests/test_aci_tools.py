from __future__ import annotations

from pathlib import Path

from tools.aci import grep, read_file, run_pytest, write_file


def test_read_write_and_grep(tmp_path: Path) -> None:
    target = tmp_path / "sample.txt"
    write_result = write_file({"path": target, "text": "hello world"})
    assert write_result["text"] == "OK"

    read_result = read_file({"path": target})
    assert read_result["text"] == "hello world"

    grep_result = grep({"pattern": "hello", "root": tmp_path})
    assert str(target) in grep_result["text"]


def test_run_pytest(tmp_path: Path, monkeypatch) -> None:
    test_file = tmp_path / "test_demo.py"
    test_file.write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    result = run_pytest({"args": "-q"})
    assert "1 passed" in result["text"]
