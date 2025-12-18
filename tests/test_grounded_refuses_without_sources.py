from behaviors.llama_generate import LlamaGenerate


def test_grounded_refuses_without_sources() -> None:
    behavior = LlamaGenerate()
    ctx = {
        "text": "Explain zero trust security.",
        "data": {
            "question": "Explain zero trust security.",
            "goal_flags": ["grounded", "cited"],
            "llama_profile": "research_plan",
            "passages": [
                {
                    "text": "Zero trust assumes every request could be malicious.",
                    "meta": {"title": "Internal Draft"},
                }
            ],
        },
    }

    result = behavior.run(ctx)

    assert result["ok"] is False
    logs = result.get("logs", [])
    assert any("no citable sources" in str(msg) for msg in logs)
    assert "llama_output" not in ctx["data"]
    assert "grounded" not in result.get("effects", [])
