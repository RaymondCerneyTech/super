from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pytest

from core.interpreter import Interpreter
from core.registry import BehaviorRegistry


@pytest.fixture()
def registry() -> BehaviorRegistry:
    reg = BehaviorRegistry()
    reg.discover().load_meta()
    return reg


@pytest.fixture()
def interpreter(registry: BehaviorRegistry, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Interpreter:
    monkeypatch.setenv("SUPER_INDEX_DIR", str(tmp_path / "indexes"))
    return Interpreter(registry)


def test_ingest_retrieve_answer_verbose(monkeypatch: pytest.MonkeyPatch, interpreter: Interpreter) -> None:
    sample_text = (
        "Artificial intelligence systems improve automation, unlock insights, and require careful governance. "
        "Teams should evaluate data quality, model drift, and human oversight to maintain trust."
    )

    def fake_fetch(url: str) -> Tuple[str, Dict[str, str]]:
        return sample_text, {
            "url": url,
            "content_type": "text/plain",
            "parser": "text",
        }

    monkeypatch.setattr("core.fetch.fetch_url", fake_fetch)
    monkeypatch.setattr("core.fetch.is_probably_english", lambda text: True)

    ctx = {
        "data": {
            "url": "https://example.com/ai-governance",
            "tags": "ai,governance",
            "index_backend": "tfidf",
            "question": "How should teams govern AI systems responsibly?",
            "k_passages": 5,
            "max_chars": 4000,
            "verbosity": "verbose",
            "min_words": 200,
        }
    }

    ingest = interpreter.execute("ingest_web", ctx)
    assert ingest["ok"] is True
    assert ctx["data"].get("ingest_log")

    retrieval = interpreter.execute("retrieve", ctx)
    assert retrieval["ok"] is True
    assert ctx["data"]["passages"]

    aggregate = interpreter.execute("aggregate", ctx)
    assert aggregate["ok"] is True
    assert ctx["data"]["aggregated_text"]

    answer = interpreter.execute("answer_verbose", ctx)
    assert answer["ok"] is True
    effects = set(answer["effects"])
    assert "cited" in effects
    assert "grounded" in effects
    assert "verbose" in effects
    rewards = answer["rewards"]
    assert rewards.get("citation_coverage", 0.0) > 0.0
    assert rewards.get("faithfulness", 0.0) > 0.0
    word_count = ctx["data"]["word_count"]
    assert word_count >= ctx["data"]["min_words"] - 10
