from pathlib import Path
from typing import Any, Dict

from behaviors.llama_generate import LlamaGenerate


def test_llama_behavior_profile(monkeypatch, tmp_path):
    profiles_path = tmp_path / "llama_profiles.yaml"
    profiles_path.write_text(
        "default_model: test.gguf\nprofiles:\n  poem:\n    prompt: Tool sonnet about {topic}.\n    n_predict: 16\n    extra:\n      - --simple-io\n",
        encoding="utf-8",
    )
    models_root = tmp_path / "models"
    models_root.mkdir()
    (models_root / "test.gguf").write_text("placeholder", encoding="utf-8")

    monkeypatch.setenv("SUPER_LLAMA_PROFILES", str(profiles_path))
    monkeypatch.setenv("SUPER_MODELS_ROOT", str(models_root))
    monkeypatch.setenv("LLAMA_BIN", str(tmp_path / "fake_llama"))

    captured: Dict[str, Any] = {}

    def fake_run_inference(**kwargs):
        captured.update(kwargs)
        return {"stdout": "Tool line one\nTool line two", "stderr": "", "returncode": "0", "command": "llama"}

    monkeypatch.setattr("tools.llama_runner.run_inference", fake_run_inference)

    behavior = LlamaGenerate()
    ctx = {
        "text": "",
        "data": {
            "llama": {
                "profile": "poem",
                "vars": {"topic": "tools"},
                "model": str(models_root / "test.gguf"),
            }
        },
    }

    result = behavior.run(ctx)
    assert result["ok"] is True
    assert "Tool line one" in result["output"]["text"]
    assert ctx["data"]["llama_output"]["text"].startswith("Tool line one")
    assert "--simple-io" in captured["extra_args"]


    state_path = Path(".ai") / "models_state.json"
    if state_path.exists():
        state_path.unlink()

