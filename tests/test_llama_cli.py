import os
import subprocess
import sys
from pathlib import Path


def run_cli(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "main.py", *args]
    run_env = None
    if env:
        run_env = os.environ.copy()
        run_env.update(env)
    return subprocess.run(cmd, capture_output=True, text=True, check=True, env=run_env)


def test_llama_cli_runs_with_mock(tmp_path, monkeypatch):
    models_root = tmp_path / "models"
    models_root.mkdir()
    (models_root / "alpha.gguf").write_text("placeholder", encoding="utf-8")
    state_path = Path(".ai") / "models_state.json"
    if state_path.exists():
        state_path.unlink()

    env = {"SUPER_MODELS_ROOT": str(models_root)}

    monkeypatch.setattr("tools.llama_runner.find_llama_binary", lambda: Path("dummy/bin/llama"))

    def fake_run(args, capture_output=None, text=None, env=None, **kwargs):
        assert "--model" in args
        assert "alpha.gguf" in " ".join(str(a) for a in args)
        return subprocess.CompletedProcess(args, 0, stdout="LLM output text", stderr="")

    monkeypatch.setattr("tools.llama_runner.subprocess.run", fake_run)

    result = run_cli("llama", "--prompt", "Hello world", "--model", "alpha.gguf", env=env)
    assert "LLM output text" in result.stdout

    if state_path.exists():
        state_path.unlink()
