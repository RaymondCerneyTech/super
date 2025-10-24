import json
from pathlib import Path

import pytest

import main
from core.registry import BehaviorRegistry
from core.router import SimpleRouter, extract_features


def test_extract_features_stable() -> None:
    goal = "summary,compliant,formatted"
    text = "Draft contract includes the secret roadmap."
    features = extract_features(goal, text)
    repeat = extract_features(goal, text)
    assert features == repeat
    assert len(features) == 14
    assert features[0] == 1.0  # bias term


def test_plan_command_logs_jsonl(tmp_path: Path) -> None:
    log_file = tmp_path / "bandit.jsonl"
    args = [
        "plan",
        "--goal",
        "summary,compliant,formatted",
        "--text",
        "Draft contract includes the secret roadmap.",
        "--policies",
        "secret roadmap",
        "--log-file",
        str(log_file),
        "--max-expansions",
        "6",
    ]

    exit_code = main.main(args)
    assert exit_code == 0
    exit_code = main.main(args)
    assert exit_code == 0

    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        record = json.loads(line)
        assert record["goal"] == "summary,compliant,formatted"
        assert "features" in record and isinstance(record["features"], list)
        assert "chosen_cluster" in record
        assert "final_rewards" in record and isinstance(record["final_rewards"], dict)


def test_router_state_persistence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state_path = tmp_path / "router_state.json"
    monkeypatch.setenv("SUPER_ROUTER_STATE_PATH", str(state_path))

    registry = BehaviorRegistry().discover().load_meta()
    router = SimpleRouter(registry)

    ctx = {
        "text": "Governance requires oversight",
        "data": {
            "text": "Governance requires oversight",
            "k_passages": 6,
            "max_chars": 6000,
        },
    }
    goal = "cited,verbose"
    cluster = router.cluster_hint(goal, ctx)
    router.register_outcome(cluster, {"overall": 0.8})
    router.register_bandit_outcome(ctx, {"overall": 0.8})

    assert state_path.exists()
    initial_value = router.cluster_values.get(cluster)

    router_again = SimpleRouter(registry)
    loaded_value = router_again.cluster_values.get(cluster)
    assert loaded_value is not None
    assert loaded_value == pytest.approx(initial_value)

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert "bandit" in payload
    assert payload["bandit"]["arms"]
    assert any(abs(value) > 0 for value in payload["bandit"]["b"][cluster])
