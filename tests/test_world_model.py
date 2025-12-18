import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from behaviors.mpc_plan import MPCPlan
from behaviors.log_rollouts import LogRollouts
from core.registry import BehaviorRegistry
from core.world_model import compact_state
from scripts.train_dynamics import train_model


def _write_rollouts(tmp_path: Path) -> Path:
    rollouts_path = tmp_path / "rollouts.jsonl"
    states = [
        {
            "meaning": "analyze_and_comment",
            "has_sources": False,
            "k_passages": 5,
            "fresh_days": 7,
            "tokens_out": 0,
            "profile": "research_plan",
        },
        {
            "meaning": "analyze_and_comment",
            "has_sources": True,
            "k_passages": 12,
            "fresh_days": 30,
            "tokens_out": 400,
            "profile": "research_plan",
        },
    ]
    actions = [
        {"name": "retrieve", "args": {"k_passages": 5}},
        {"name": "llama_generate", "args": {"profile": "research_plan"}},
    ]
    next_states = [
        {
            "meaning": "analyze_and_comment",
            "has_sources": True,
            "k_passages": 5,
            "fresh_days": 7,
            "tokens_out": 0,
            "profile": "research_plan",
        },
        {
            "meaning": "analyze_and_comment",
            "has_sources": True,
            "k_passages": 12,
            "fresh_days": 30,
            "tokens_out": 380,
            "profile": "research_plan",
        },
    ]
    with rollouts_path.open("w", encoding="utf-8") as handle:
        for t, (s, a, sp) in enumerate(zip(states, actions, next_states)):
            handle.write(json.dumps({"t": t, "s": s, "a": a, "s_prime": sp, "r": 0.5}) + "\n")
    return rollouts_path


def test_dynamics_training_and_mpc(monkeypatch, tmp_path):
    import random

    rollouts_path = _write_rollouts(tmp_path)
    model_path = tmp_path / "ensemble.pt"
    train_model(rollouts_path, model_path, steps=50, batch_size=2, seed=42)
    assert model_path.exists()

    registry = BehaviorRegistry().discover().load_meta()
    mpc = registry.get("mpc_plan")

    ctx = {
        "data": {
            "meaning": "analyze_and_comment",
            "passages": [{"text": "Prime numbers are distributed irregularly."}],
            "k_passages": 5,
            "fresh_days": 7,
            "llama_profile": "research_plan",
            "dynamics_model_path": str(model_path),
        }
    }
    random.seed(0)
    result = mpc.run(ctx)
    action = ctx["data"].get("next_action")
    assert result["ok"] is True
    assert isinstance(action, dict)
    assert action.get("name") in {"retrieve", "aggregate", "llama_generate", "format"}


def test_log_rollouts_records(monkeypatch, tmp_path):
    target_path = tmp_path / "rollouts.jsonl"
    monkeypatch.setattr("behaviors.log_rollouts.ROLL_OUT_PATH", target_path)
    log_behavior = LogRollouts()
    ctx = {"data": {}}
    ctx["data"].update(
        {
            "_prev_state": compact_state({"meaning": "analyze_and_comment"}),
            "_last_action": {"name": "retrieve", "args": {"k_passages": 5}},
            "passages": [{"text": "Example passage."}],
            "llama_output": {"text": "Answer with [1] citation."},
        }
    )
    result = log_behavior.run(ctx)
    assert result["ok"] is True
    assert target_path.exists()
