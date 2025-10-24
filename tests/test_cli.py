import json
import os
import subprocess
import sys


def run_cli(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "main.py", *args]
    run_env = None
    if env:
        run_env = os.environ.copy()
        run_env.update(env)
    return subprocess.run(cmd, capture_output=True, text=True, check=True, env=run_env)


def test_cli_help_lists_commands():
    result = run_cli("help")
    output = result.stdout.lower()
    assert "summarize" in output
    assert "format" in output
    assert "optimize" in output
    assert "check" in output


def test_cli_summarize_extractive():
    text = "Artificial intelligence enables automation and better decisions."
    result = run_cli(
        "summarize",
        "--text",
        text,
        "--strategy",
        "extractive",
        "--max-words",
        "10",
    )
    output = result.stdout
    assert "Summary:" in output
    assert "Reward" in output


def test_cli_policy_check():
    text = "Share the secret roadmap with partners."
    result = run_cli(
        "check",
        "--text",
        text,
        "--policies",
        "secret roadmap",
    )
    output = result.stdout
    assert "[REDACTED]" in output
    assert "Violations" in output


def test_cli_root_help_overview():
    result = subprocess.run([sys.executable, "main.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    output = result.stdout.lower()
    assert "available commands" in output
    assert "summarize --text" in output
    assert "format --text" in output

def test_cli_plan_explain():
    text = "This document needs a quick summary before posting."
    result = run_cli(
        "plan",
        "--goal",
        "summary",
        "--text",
        text,
        "--explain",
        "--max-expansions",
        "5",
    )
    output = result.stdout.lower()
    assert "plan:" in output
    assert "summary" in output
    assert "final reward" in output


def test_cli_ingest_respects_config(tmp_path):
    config_path = tmp_path / "super.yaml"
    config_path.write_text("ingest:\n  index_backend: hnsw\n  tags: quickstart\n", encoding="utf-8")
    sample = tmp_path / "sample.txt"
    sample.write_text("Hello knowledge", encoding="utf-8")
    env = {"SUPER_INDEX_DIR": str(tmp_path / "indexes")}
    result = run_cli("--config", str(config_path), "ingest", "--path", str(sample), env=env)
    output = result.stdout
    assert "Docs ingested: 1 (backend=hnsw)" in output

