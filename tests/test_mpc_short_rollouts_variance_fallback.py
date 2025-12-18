from pathlib import Path
from types import SimpleNamespace

from behaviors.mpc_plan import MPCPlan


def test_mpc_short_rollouts_variance_fallback(monkeypatch, tmp_path) -> None:
    mpc = MPCPlan()
    monkeypatch.setattr("behaviors.mpc_plan.torch", object(), raising=False)

    fake_path = Path(tmp_path) / "ensemble.pt"
    fake_ensemble = SimpleNamespace(_loaded_path=fake_path)
    mpc._ensemble = fake_ensemble
    mpc._action_candidates = [{"name": "llama_generate", "args": {}}]

    monkeypatch.setattr(mpc, "_evaluate_sequence", lambda state_vec, sequence, penalty: (0.5, 9.0), raising=False)

    ctx = {
        "data": {
            "meaning": "analyze_and_comment",
            "dynamics_model_path": str(fake_path),
            "k_passages": 5,
            "passages": [],
            "mpc_rollout_candidates": 1,
            "mpc_rollout_horizon": 1,
            "mpc_variance_threshold": 2.5,
        }
    }

    result = mpc.run(ctx)

    assert result["ok"] is True
    assert ctx["data"]["next_action"]["name"] == "retrieve"
    assert ctx["data"]["mpc_mode"] == "heuristic"
    assert any("variance" in log for log in result.get("logs", []))
