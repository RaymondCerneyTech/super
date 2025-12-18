import importlib
import json
from pathlib import Path

from core.interfaces import Context


def test_bundle_selection_and_reward_record(tmp_path, monkeypatch):
    state_path = tmp_path / "bundle.json"
    monkeypatch.setenv("SUPER_BUNDLE_STATE", str(state_path))
    module = importlib.import_module("core.meta_controller")
    meta_controller = importlib.reload(module)
    monkeypatch.setattr(meta_controller.llama_profiles, "list_profiles", lambda: [("research_plan", ""), ("math_solver", "")])

    ctx: Context = {
        "text": "Summarize the latest research plan.",
        "data": {"text": "Summarize the latest research plan."},
        "router": {"goal_text": "summary,grounded"},
    }
    bundle = meta_controller.choose_bundle(ctx, planners=["planner_react", "planner_tot", "planner_fp"], judge_bundle=["meta_judge"])

    assert bundle["planners"]
    assert ctx["router"]["bundle_arm"]
    assert ctx["data"]["model_bundle"]

    meta_controller.record_bundle_outcome(ctx, {"overall": 0.8, "verifier": 0.9})
    assert state_path.exists()
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert ctx["router"]["bundle_cue"] in payload
    log_path = Path("logs") / "bundles.jsonl"
    assert log_path.exists()
    assert log_path.read_text(encoding="utf-8").strip()
