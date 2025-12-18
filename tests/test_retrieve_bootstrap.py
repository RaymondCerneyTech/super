import shutil
from pathlib import Path

import pytest

from behaviors.retrieve import Retrieve
from core import index as index_mod


def _reset_index_cache(tmp_path: Path) -> None:
    if (tmp_path / "indexes").exists():
        shutil.rmtree(tmp_path / "indexes")
    index_mod._INDEX_CACHE.clear()


@pytest.fixture()
def isolated_index(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPER_INDEX_DIR", str(tmp_path / "indexes"))
    _reset_index_cache(tmp_path)
    yield
    _reset_index_cache(tmp_path)


def test_retrieve_bootstrap_text(isolated_index) -> None:
    retrieve = Retrieve()
    ctx = {
        "data": {
            "question": "Summarize current approaches to distributed systems testing.",
            "bootstrap_texts": [
                "Distributed systems testing focuses on fault injection, chaos engineering, and observability."
            ],
            "auto_bootstrap": True,
            "index_backend": "hnsw",
        }
    }
    result = retrieve.run(ctx)
    assert result["ok"] is True
    passages = ctx["data"].get("passages", [])
    assert passages
    # running again should not duplicate entries
    second = retrieve.run(ctx)
    assert second["ok"] is True
    assert len(ctx["data"].get("passages", [])) >= 1


def test_retrieve_builtin_seed(isolated_index) -> None:
    retrieve = Retrieve()
    ctx = {
        "data": {
            "question": "What tactics are researchers using on the Riemann Hypothesis today?",
            "auto_bootstrap": True,
            "index_backend": "hnsw",
        }
    }
    result = retrieve.run(ctx)
    assert result["ok"] is True
    passages = ctx["data"].get("passages", [])
    assert passages
    # Built-in seed should mark origin
    assert any("builtin_seed" in (p.get("meta", {}).get("origin", "") or "") for p in passages)
