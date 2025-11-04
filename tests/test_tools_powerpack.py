from __future__ import annotations

from pathlib import Path

from tools.aci import append_file, read_file, write_file, grep_repo
from tools.data_utils import csv_summary, table_detect
from tools.web import html_to_text, web_get


def test_web_tools_local_html(tmp_path: Path) -> None:
    html_content = "<html><body><h1>Hello</h1><a href='https://example.com'>Example</a></body></html>"
    html_path = tmp_path / "sample.html"
    html_path.write_text(html_content, encoding="utf-8")

    result = web_get({"url": html_path.as_uri()})
    assert result["status"] == 200
    assert "Hello" in result["text"]

    converted = html_to_text({"html": html_content})
    assert "<" not in converted["text"]
    assert "Example" in converted["text"]


def test_data_tools_table_summary() -> None:
    text_block = "Name,Score\nAlice,10\nBob,20"
    table = table_detect({"text": text_block})
    assert "Name,Score" in table["text"]

    summary = csv_summary({"csv": table["text"]})
    assert "rows: 2" in summary["text"]
    assert "Score" in summary["text"]


def test_aci_round_trip(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.txt"
    write_resp = write_file({"path": str(file_path), "text": "Line one\n"})
    assert write_resp["text"] == "OK"

    append_resp = append_file({"path": str(file_path), "text": "Line two\n"})
    assert append_resp["text"] == "OK"

    read_resp = read_file({"path": str(file_path)})
    assert "Line two" in read_resp["text"]

    grep_resp = grep_repo({"root": str(tmp_path), "pattern": "Line two"})
    assert str(file_path) in grep_resp["text"]

