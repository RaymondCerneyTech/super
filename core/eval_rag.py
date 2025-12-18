from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


@dataclass
class RagRecord:
    question: str
    answer: str
    contexts: List[str]
    expected_answer: str


def load_rag_dataset(path: Path) -> List[RagRecord]:
    text = path.read_text(encoding="utf-8")
    payload: List[Dict[str, Any]]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    else:
        if isinstance(parsed, dict):
            parsed = [parsed]
        elif not isinstance(parsed, list):
            raise ValueError("Dataset must be a JSON list, dict, or JSONL file.")

    records: List[RagRecord] = []
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        question = str(entry.get("question") or entry.get("prompt") or "").strip()
        answer = str(entry.get("answer") or "").strip()
        contexts_raw = entry.get("contexts") or entry.get("passages") or []
        if isinstance(contexts_raw, str):
            contexts = [contexts_raw]
        elif isinstance(contexts_raw, Iterable):
            contexts = [str(item) for item in contexts_raw]
        else:
            contexts = []
        expected = str(entry.get("expected_answer") or entry.get("reference") or question).strip()
        records.append(RagRecord(question=question, answer=answer, contexts=contexts, expected_answer=expected))
    if not records:
        raise ValueError("No evaluation records found in dataset.")
    return records


def _token_overlap(answer: str, context: str, min_token_length: int = 5) -> bool:
    answer_lower = answer.lower()
    for token in context.split():
        token = token.lower()
        if len(token) >= min_token_length and token in answer_lower:
            return True
    return False


def score_record(record: RagRecord) -> Dict[str, float]:
    answer_lower = record.answer.lower()
    expected_lower = record.expected_answer.lower()
    contexts = record.contexts

    if contexts:
        overlaps = [_token_overlap(record.answer, ctx) for ctx in contexts]
        faithfulness = sum(overlaps) / len(overlaps)
        context_relevance = sum(1 for ctx in contexts if _token_overlap(record.question, ctx, min_token_length=4)) / len(contexts)
    else:
        faithfulness = 0.0
        context_relevance = 0.0

    if expected_lower:
        answer_relevance = 1.0 if expected_lower in answer_lower else 0.0
    else:
        answer_relevance = 0.5

    return {
        "faithfulness": faithfulness,
        "context_relevance": context_relevance,
        "answer_relevance": answer_relevance,
    }


def aggregate_metrics(per_item: List[Dict[str, float]]) -> Dict[str, float]:
    metrics = {
        "faithfulness": 0.0,
        "context_relevance": 0.0,
        "answer_relevance": 0.0,
    }
    if not per_item:
        return metrics
    for entry in per_item:
        for key in metrics:
            metrics[key] += float(entry.get(key, 0.0))
    count = float(len(per_item))
    for key in metrics:
        metrics[key] = round(metrics[key] / count, 4)
    return metrics


def run_rag_evaluation(
    dataset_path: Path,
    output_dir: Path,
) -> Tuple[Dict[str, float], Path]:
    records = load_rag_dataset(dataset_path)
    per_item_scores: List[Dict[str, float]] = [score_record(record) for record in records]
    metrics = aggregate_metrics(per_item_scores)

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    eval_root = output_dir / timestamp
    eval_root.mkdir(parents=True, exist_ok=True)

    report = {
        "dataset": str(dataset_path),
        "metrics": metrics,
        "records": [
            {
                "question": record.question,
                "answer": record.answer,
                "contexts": record.contexts,
                "expected_answer": record.expected_answer,
                "scores": score,
            }
            for record, score in zip(records, per_item_scores)
        ],
    }
    report_path = eval_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    markdown_lines = [
        "# RAG Evaluation Report",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Records: {len(records)}",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for key, value in metrics.items():
        markdown_lines.append(f"| {key} | {value:.4f} |")
    (eval_root / "report.md").write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    return metrics, eval_root


__all__ = ["run_rag_evaluation", "load_rag_dataset"]

