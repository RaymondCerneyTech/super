from __future__ import annotations

import json
from pathlib import Path

from core.interpreter import Interpreter
from core.registry import BehaviorRegistry
from core import reflections


class DummyTools:
    @staticmethod
    def failing_summarize(payload):
        return {"text": "", "quality_gain": 0.0, "faithfulness": 1.0, "notes": "no summary"}

    @staticmethod
    def failing_numbers(payload):
        return {"text": payload.get("text", ""), "quality_gain": 0.0, "faithfulness": 0.0, "notes": "numbers changed"}

    @staticmethod
    def success_summarize(payload):
        text = payload.get("text", "")
        return {"text": "- " + text, "quality_gain": 0.2, "faithfulness": 1.0, "notes": "summary ok"}

    @staticmethod
    def success_numbers(payload):
        return {"text": payload.get("text", ""), "quality_gain": 0.0, "faithfulness": 1.0, "notes": "numbers ok"}


FAILING_TOOLS = {
    "summarize_bullets": {"name": "summarize_bullets", "affordances": ["summarize", "compose"], "fn": DummyTools.failing_summarize},
    "numbers_guard": {"name": "numbers_guard", "affordances": ["numbers", "guard"], "fn": DummyTools.failing_numbers},
}

SUCCESS_TOOLS = {
    "summarize_bullets": {"name": "summarize_bullets", "affordances": ["summarize", "compose"], "fn": DummyTools.success_summarize},
    "numbers_guard": {"name": "numbers_guard", "affordances": ["numbers", "guard"], "fn": DummyTools.success_numbers},
}


def test_reflections_help_second_run(monkeypatch, tmp_path):
    reflections_path = tmp_path / "reflections.jsonl"
    monkeypatch.setattr(reflections, "REFLECTIONS_PATH", reflections_path)
    registry = BehaviorRegistry().discover().load_meta()
    interpreter = Interpreter(registry)

    prompt = "Summarize the 3.14 result precisely."
    base_ctx = {
        "data": {
            "goal": prompt,
            "text": prompt,
            "prompt": prompt,
            "draft": prompt,
            "invariants": {"preserve_numbers": True, "source_text": prompt},
            "max_iters": 2,
        }
    }

    ctx_fail = json.loads(json.dumps(base_ctx))
    ctx_fail["data"]["tools_registry"] = FAILING_TOOLS
    result_fail = interpreter.execute("deep_loop", ctx_fail)
    assert reflections_path.exists()
    lines = reflections_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines

    ctx_success = json.loads(json.dumps(base_ctx))
    ctx_success["data"]["tools_registry"] = SUCCESS_TOOLS
    result_success = interpreter.execute("deep_loop", ctx_success)
    assert result_success["ok"]
    episodes = result_success["output"]["episodes"]
    hints = episodes[0]["step"].get("hints", [])
    assert hints, "expected reflection hints on second run"
    assert any("underperformed" in hint for hint in hints)
