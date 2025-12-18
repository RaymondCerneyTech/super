from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from core.interpreter import Interpreter
from core.registry import BehaviorRegistry


def test_retrieval_prefers_recent(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPER_INDEX_DIR", str(tmp_path / "idx"))

    registry = BehaviorRegistry().discover().load_meta()
    interpreter = Interpreter(registry)

    now = datetime.now(timezone.utc)
    old_ts = (now - timedelta(days=90)).isoformat()
    new_ts = (now - timedelta(days=5)).isoformat()

    def fake_fetch(url: str):
        if "old" in url:
            return (
                "Legacy policy guidance emphasizing manual oversight.",
                {"url": url, "content_type": "text/plain", "parser": "text", "published_ts": old_ts},
            )
        return (
            "Recent governance update covering automated monitoring and audits.",
            {"url": url, "content_type": "text/plain", "parser": "text", "published_ts": new_ts},
        )

    with patch("core.fetch.fetch_url", fake_fetch), patch("core.fetch.is_probably_english", lambda text: True):
        interpreter.execute(
            "ingest_web",
            {"data": {"url": "https://example.com/old", "index_backend": "hnsw", "tags": "governance"}},
        )
        interpreter.execute(
            "ingest_web",
            {"data": {"url": "https://example.com/new", "index_backend": "hnsw", "tags": "governance"}},
        )

    ctx = {
        "data": {
            "question": "What are the latest AI governance practices?",
            "index_backend": "hnsw",
            "k_passages": 3,
            "max_chars": 2000,
            "tags": "governance",
            "fresh_days": 30,
        }
    }

    result = interpreter.execute("retrieve", ctx)
    assert result["ok"] is True
    passages = ctx["data"]["passages"]
    assert passages
    first = passages[0]
    assert "https://example.com/new" in (first.get("meta", {}).get("url") or "")
    assert first.get("recency", 0.0) >= 0.5
    old_entry = next((p for p in passages if "old" in (p.get("meta", {}).get("url") or "")), None)
    assert old_entry is not None
    assert first.get("recency", 0.0) > old_entry.get("recency", 0.0)
