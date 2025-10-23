import pytest

from core.interpreter import Interpreter
from core.registry import BehaviorRegistry
from core.router import SimpleRouter


@pytest.fixture(scope="module")
def registry() -> BehaviorRegistry:
    reg = BehaviorRegistry()
    reg.discover().load_meta()
    return reg


@pytest.fixture
def interpreter(registry: BehaviorRegistry) -> Interpreter:
    return Interpreter(registry)


def test_registry_discovers_summarize(registry: BehaviorRegistry) -> None:
    names = registry.list()
    assert "summarize" in names

    meta = registry.meta("summarize")
    assert isinstance(meta, dict)
    assert meta.get("name") == "summarize"


def test_interpreter_executes_and_writes_output(interpreter: Interpreter) -> None:
    ctx = {"data": {"text": "one two three four five six", "max_words": 3}}
    result = interpreter.execute("summarize", ctx)

    assert result["ok"] is True
    assert "summary" in ctx["data"]
    assert len(ctx["data"]["summary"].split()) <= 3


def test_reward_aggregates_checks(interpreter: Interpreter) -> None:
    ctx = {"data": {"text": "alpha beta gamma delta", "max_words": 3}}
    result = interpreter.execute("summarize", ctx)

    assert result["reward"] >= 1.0
    assert "length_leq" in result["checks"]
    assert result["checks"]["length_leq"] == 1.0


def test_router_prefers_summarize_for_long_text(registry: BehaviorRegistry) -> None:
    router = SimpleRouter(registry)
    long_text = " ".join(["This summary aims to condense important points."] * 10)
    ctx = {"text": long_text, "data": {"text": long_text}}

    choice = router.choose(ctx)
    assert choice == "summarize"


def test_router_prefers_rewrite_style_for_professional_tone(registry: BehaviorRegistry) -> None:
    router = SimpleRouter(registry)
    text = "Please rewrite this message in a professional tone with a polished style."
    ctx = {"text": text, "data": {"text": text}}

    choice = router.choose(ctx)
    assert choice == "rewrite_style"
