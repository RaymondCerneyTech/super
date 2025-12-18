import argparse
import json

from main import command_eval_rag


def test_ragas_report_contains_faithfulness(tmp_path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "question": "What is AI?",
                        "answer": "Artificial intelligence systems learn patterns.",
                        "contexts": ["Artificial intelligence systems learn patterns from data."],
                        "expected_answer": "learn patterns",
                    }
                ),
                json.dumps(
                    {
                        "question": "Name a benefit of automation.",
                        "answer": "Automation improves efficiency across teams.",
                        "contexts": ["Automation improves efficiency and reduces manual errors."],
                        "expected_answer": "improves efficiency",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "reports"
    args = argparse.Namespace(
        dataset=str(dataset),
        output=str(output_dir),
        faithfulness_threshold=0.0,
    )

    command_eval_rag(args)

    generated = sorted(output_dir.iterdir())
    assert generated, "Expected evaluation output directory"
    report_path = generated[-1] / "report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert "faithfulness" in payload["metrics"]
    assert payload["metrics"]["faithfulness"] >= 0.0
