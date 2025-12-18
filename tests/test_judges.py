import json
from pathlib import Path
from typing import Dict

import pytest

from judges import meta_judge


def test_consistency_judge_majority_weight():
    candidates = [
        {"output": {"final": "Same answer"}},
        {"output": {"final": "Same answer"}},
        {"output": {"final": "Different"}},
    ]
    scores = meta_judge.consistency_judge(candidates)
    assert scores[0] == scores[1]
    assert scores[0] > scores[2]


def test_llm_pairwise_judge_fallback(monkeypatch):
    monkeypatch.delenv("LLM_JUDGE_MODEL", raising=False)
    candidates = [
        {"effects": ["compliant"], "rewards": {"overall": 0.9}},
        {"effects": [], "rewards": {"overall": 0.1}},
    ]
    winner, rationale = meta_judge.llm_pairwise_judge(candidates[0], candidates[1])
    assert winner == 0
    assert rationale


def test_llm_pairwise_judge_with_runner(monkeypatch, tmp_path):
    model_path = tmp_path / "judge.bin"
    model_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("LLM_JUDGE_MODEL", str(model_path))

    def fake_run_inference(**kwargs):
        payload = {"winner": "B", "rationale": "Answer B cites more sources"}
        return {"stdout": json.dumps(payload), "returncode": "0"}

    monkeypatch.setattr("tools.llama_runner.run_inference", fake_run_inference)
    a = {"rewards": {"overall": 0.4}, "output": {"final": "Option A"}}
    b = {"rewards": {"overall": 0.6}, "output": {"final": "Option B"}}
    winner, rationale = meta_judge.llm_pairwise_judge(a, b)
    assert winner == 1
    assert "B" in rationale


def test_aggregate_includes_llm_details(monkeypatch):
    def fake_pairwise(a: Dict, b: Dict, rubric: str = ""):
        return (0, "A wins")

    monkeypatch.setattr(meta_judge, "llm_pairwise_judge", fake_pairwise)
    candidates = [
        {"effects": ["compliant"], "rewards": {"overall": 0.8}, "output": {"final": "First"}},
        {"effects": [], "rewards": {"overall": 0.2}, "output": {"final": "Second"}},
    ]
    best_idx, details = meta_judge.aggregate(candidates)
    assert best_idx == 0
    assert details["llm"] >= 0.5
    assert details["llm_details"]
