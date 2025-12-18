from pathlib import Path

from core.registry import BehaviorRegistry


def test_hypothesis_lab_generates_ranked_hypotheses(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    registry = BehaviorRegistry().discover().load_meta()
    behavior = registry.get("hypothesis_lab")
    ctx = {
        "data": {
            "question": "How can we reduce flaky tests by 50 percent within a quarter?",
            "hypothesis_count": 3,
            "enable_evolution": True,
            "hypothesis_verifier": {
                "type": "unit_tests_pass",
                "report": {"passed": 4, "total": 4},
            },
        }
    }

    result = behavior.run(ctx)
    assert result["ok"] is True
    hypotheses = ctx["data"]["hypotheses"]
    assert len(hypotheses) >= 3
    assert hypotheses[0]["rating"] >= hypotheses[-1]["rating"]
    review_path = Path(".ai") / "meta" / "reviewer.json"
    assert review_path.exists()
    review = ctx["data"]["hypothesis_review"]
    assert review["champion"] == hypotheses[0]["id"]
