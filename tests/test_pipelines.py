from __future__ import annotations

import json
from pathlib import Path

from core import credit_ledger
from core.interpreter import Interpreter
from core.registry import BehaviorRegistry
from tools.registry import TOOLS


def test_deep_loop_runs_summarize_pipeline(monkeypatch, tmp_path) -> None:
    ledger_path = tmp_path / "ledger_tool_calls.jsonl"
    monkeypatch.setattr(credit_ledger, "LEDGER_PATH", ledger_path)

    prompt = "Summarize this 3.14 finding into clear bullet points."
    registry = BehaviorRegistry().discover().load_meta()
    interpreter = Interpreter(registry)
    ctx = {
        "data": {
            "goal": prompt,
            "text": prompt,
            "prompt": prompt,
            "draft": prompt,
            "tools_registry": TOOLS,
            "invariants": {"preserve_numbers": True, "source_text": prompt},
            "max_iters": 4,
        }
    }

    result = interpreter.execute("deep_loop", ctx)
    partial = result["output"]["partial"]["text"]

    assert partial.strip().startswith("-")
    assert "3.14" in partial
    assert ledger_path.exists()

    lines = ledger_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2
    last_two = [json.loads(line) for line in lines[-2:]]
    tool_names = [entry["tool"] for entry in last_two]
    assert tool_names == ["summarize_bullets", "numbers_guard"]
    assert last_two[-1]["faithfulness"] == 1.0
