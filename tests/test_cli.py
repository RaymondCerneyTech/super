import json
import subprocess
import sys


def run_cli(*args: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "main.py", *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


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
