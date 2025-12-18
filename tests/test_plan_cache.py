from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from core.plan_cache import PlanCache, shared_plan_cache
from core.planner import plan, normalise_goal_flags
from core.registry import BehaviorRegistry
from core.interpreter import Interpreter


def test_plan_cache_basic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUPER_PLAN_CACHE_PATH", str(tmp_path / "plans.json"))
    cache = PlanCache()
    key = cache.build_key(["cited", "verbose"], backend="hnsw", verbosity="verbose")
    assert cache.lookup(key) is None
    cache.store(key, [("retrieve", {}), ("answer_verbose", {})])
    restored = cache.lookup(key)
    assert restored == ["retrieve", "answer_verbose"]


def test_plan_cache_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUPER_PLAN_CACHE_PATH", str(tmp_path / "plans.json"))
    shared_plan_cache().clear()

    registry = BehaviorRegistry().discover().load_meta()
    interpreter = Interpreter(registry)
    ctx = {
        "data": {
            "text": "Draft contract includes the secret roadmap.",
            "max_words": 60,
            "max_sentences": 3,
            "summary_strategy": "extractive",
        }
    }
    goal_flags = normalise_goal_flags("summary")
    result_first = plan(goal_flags, ctx, registry, interpreter, cluster_bias="analytic", max_expansions=10)
    assert result_first["steps"]
    shared_plan_cache_instance = shared_plan_cache()
    meaning = result_first.get("ctx", {}).get("data", {}).get("meaning")
    first_key = shared_plan_cache_instance.build_key(
        goal_flags,
        backend="hnsw",
        verbosity="normal",
        meaning=meaning,
    )
    assert shared_plan_cache_instance.lookup(first_key) is not None

    with patch("core.planner.shared_plan_cache", return_value=shared_plan_cache_instance):
        result_second = plan(goal_flags, ctx, registry, interpreter, cluster_bias="analytic", max_expansions=10)
    assert result_second["expansions"] == 0
    assert result_second["goal_satisfied"]
