import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict


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


def test_cli_models_list_and_select(tmp_path):
    models_root = tmp_path / "models"
    models_root.mkdir()
    (models_root / "alpha.gguf").write_text("placeholder", encoding="utf-8")
    (models_root / "zeta").mkdir()
    state_path = Path(".ai") / "models_state.json"
    if state_path.exists():
        state_path.unlink()

    env = {"SUPER_MODELS_ROOT": str(models_root)}

    result_list = run_cli("models", "--list", env=env)
    assert "alpha.gguf" in result_list.stdout

    result_select = run_cli("models", "--select", "1", env=env)
    assert "Selected model #1" in result_select.stdout

    result_show = run_cli("models", "--show", env=env)
    assert "Active model" in result_show.stdout

    if state_path.exists():
        state_path.unlink()



def test_cli_llama_profile(monkeypatch, tmp_path, capsys):
    profiles_path = tmp_path / "llama_profiles.yaml"
    profiles_path.write_text(
        "default_model: test.gguf\nprofiles:\n  poem:\n    prompt: Write a poem about {topic}.\n    n_predict: 32\n    temperature: 0.4\n    extra:\n      - --simple-io\n",
        encoding="utf-8",
    )
    models_root = tmp_path / "models"
    models_root.mkdir()
    (models_root / "test.gguf").write_text("placeholder", encoding="utf-8")
    state_path = Path(".ai") / "models_state.json"
    if state_path.exists():
        state_path.unlink()

    monkeypatch.setenv("SUPER_LLAMA_PROFILES", str(profiles_path))
    monkeypatch.setenv("SUPER_MODELS_ROOT", str(models_root))
    monkeypatch.setenv("LLAMA_BIN", str(tmp_path / "fake_llama"))

    captured: Dict[str, Any] = {}

    def fake_run_inference(**kwargs):
        captured.update(kwargs)
        return {"stdout": "poem output", "stderr": "", "returncode": "0", "command": "llama"}

    monkeypatch.setattr("tools.llama_runner.run_inference", fake_run_inference)

    from main import main as cli_main

    exit_code = cli_main(
        [
            "llama",
            "--profile",
            "poem",
            "--model",
            str(models_root / "test.gguf"),
            "--var",
            "topic=tools",
        ]
    )
    captured_stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "poem output" in captured_stdout
    assert "tools" in captured["prompt"]
    assert "--simple-io" in captured["extra_args"]

    if state_path.exists():
        state_path.unlink()


def test_cli_plan_injects_llama_payload(monkeypatch, capsys):
    captured: Dict[str, Any] = {}

    def fake_plan(goal_flags, ctx, registry, interpreter, **kwargs):
        captured["ctx"] = ctx
        return {
            "steps": [
                ("llama_generate", {"rewards": {"overall": 0.5}, "effects": ["llm_output"], "rationale": {}})
            ],
            "ctx": ctx,
            "goal_satisfied": True,
            "remaining_flags": [],
        }

    monkeypatch.setattr("core.planner.plan", fake_plan)

    from main import main as cli_main

    exit_code = cli_main(
        [
            "plan",
            "--goal",
            "llm_output,formatted",
            "--text",
            "Analyze and comment on emerging research directions.",
            "--llama-profile",
            "research_plan",
            "--llama-var",
            "topic=Riemann Hypothesis",
        ]
    )
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Plan:" in output
    assert "llama_generate" in output
    ctx = captured["ctx"]
    assert isinstance(ctx, dict)
    data = ctx.get("data", {})
    assert isinstance(data, dict)
    payload = data.get("llama")
    assert isinstance(payload, dict)
    assert payload.get("profile") == "research_plan"
    vars_payload = payload.get("vars")
    assert isinstance(vars_payload, dict)
    assert vars_payload.get("topic") == "Riemann Hypothesis"


def test_cli_plan_research_profile_blueprint(monkeypatch, tmp_path, capsys):
    key = "flags=formatted,llm_output|backend=tfidf|verbosity=normal|tags=|meaning=analyze_and_comment|pipeline="
    blueprint_payload = {
        key: {
            "behaviors": ["meaning_infer", "retrieve", "llama_generate", "document_formatting"],
            "created_ts": time.time(),
        }
    }
    blueprint_path = tmp_path / "blueprints.json"
    blueprint_path.write_text(json.dumps(blueprint_payload), encoding="utf-8")
    monkeypatch.setenv("SUPER_PLAN_CACHE_PATH", str(blueprint_path))

    import core.plan_cache as plan_cache

    monkeypatch.setattr(plan_cache, "_SHARED_CACHE", None)

    profiles_path = tmp_path / "llama_profiles.yaml"
    profiles_path.write_text(
        "default_model: model.gguf\n"
        "profiles:\n"
        "  research_plan:\n"
        "    prompt: \"Research memo for {topic}.\"\n"
        "    n_predict: 64\n"
        "    temperature: 0.2\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SUPER_LLAMA_PROFILES", str(profiles_path))

    model_path = tmp_path / "model.gguf"
    model_path.write_text("stub", encoding="utf-8")
    monkeypatch.setattr("core.models.resolve_model_path", lambda identifier=None: model_path)

    captured_call: Dict[str, Any] = {}

    def fake_run_inference(**kwargs):
        captured_call.update(kwargs)
        return {"stdout": "Memo output", "stderr": "", "returncode": "0", "command": "llama"}

    monkeypatch.setattr("tools.llama_runner.run_inference", fake_run_inference)
    monkeypatch.setenv("SUPER_INDEX_DIR", str(tmp_path / "indexes"))

    from main import main as cli_main

    exit_code = cli_main(
        [
            "plan",
            "--goal",
            "llm_output,formatted",
            "--text",
            "Analyze and comment on current strategies for the Riemann Hypothesis research landscape.",
            "--llama-profile",
            "research_plan",
            "--llama-var",
            "topic=Riemann Hypothesis strategies",
            "--max-expansions",
            "4",
        ]
    )
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Plan:" in output
    assert "llama_generate" in output
    assert captured_call.get("model") == model_path
    assert captured_call.get("prompt") == "Analyze and comment on current strategies for the Riemann Hypothesis research landscape."
    n_predict = captured_call.get("n_predict")
    if n_predict is not None:
        assert float(n_predict) > 0
    extra_args = captured_call.get("extra_args", [])
    assert isinstance(extra_args, list)
    assert "--repeat-penalty" in extra_args
