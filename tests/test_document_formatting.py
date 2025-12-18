from behaviors.document_formatting import DocumentFormatting


def _base_data():
    return {
        "goal_flags": ["llm_output", "grounded", "cited"],
        "_llm_generated": True,
        "triangulation_met": True,
        "sources": [
            {"marker": "S1", "url": "https://example.com/a"},
            {"marker": "S2", "url": "https://example.org/b"},
        ],
        "aggregated_text": (
            "Current tactics balance computation and theory. "
            "Obstacles include limited domains. Resources include collaborations."
        ),
    }


def test_document_formatting_refuses_with_single_citation():
    formatter = DocumentFormatting()
    ctx = {
        "data": {
            **_base_data(),
            "answer": "- Insight only cites one source [S1]",
        }
    }
    result = formatter.run(ctx)
    assert result["ok"] is False
    failure_log = " ".join(result.get("logs", []))
    assert ("two sources" in failure_log) or ("three cited bullets" in failure_log)


def test_document_formatting_passes_with_valid_grounding():
    formatter = DocumentFormatting()
    ctx = {
        "data": {
            **_base_data(),
            "answer": "\n".join(
                [
                    "## Current tactics",
                    "- Combine analytic number theory with computation [S1] [S2]",
                    "- Obstacles include controlling error terms [S1] [S2]",
                    "- Resources include LMFDB collaborations [S1] [S2]",
                ]
            ),
        }
    }
    result = formatter.run(ctx)
    assert result["ok"] is True
    assert "formatted_text" in result["output"]
    assert "Analysis Notes" in result["output"]["formatted_text"]


def test_document_formatting_refuses_when_not_enough_bullets():
    formatter = DocumentFormatting()
    ctx = {
        "data": {
            **_base_data(),
            "answer": (
                "## Current tactics\n"
                "- Combine analytic number theory with computation [S1] [S2]"
            ),
        }
    }
    result = formatter.run(ctx)
    assert result["ok"] is False
    assert "three cited bullets" in " ".join(result.get("logs", []))
