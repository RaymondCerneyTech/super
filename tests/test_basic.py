from pathlib import Path

import pytest

from core.features import extract_feature_key
from core.interpreter import Interpreter
from core.learn import BanditLearner
from core.plans import run_plan
from core.registry import BehaviorRegistry
from core.router import SimpleRouter
from core.interfaces import Context


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


def test_router_picks_rewrite_style_for_professional_tone(registry: BehaviorRegistry) -> None:
    router = SimpleRouter(registry)
    ctx = {"data": {"text": "Please rewrite this memo in a professional tone"}}

    choice = router.choose(ctx)
    assert choice == "rewrite_style"


def test_bandit_bias_moves_toward_success() -> None:
    learner = BanditLearner()
    for _ in range(10):
        learner.update("summarize", 1.0, "default")

    biases = learner.adapter_biases("default")
    assert biases["summarize"] >= 0


def test_rewrite_style_invalid_tone(interpreter: Interpreter) -> None:
    ctx = {"data": {"text": "Please adjust this message.", "tone": "silly"}}

    result = interpreter.execute("rewrite_style", ctx)

    assert result["ok"] is False
    assert result["reward"] == 0.0
    assert any("tone" in log.lower() for log in result.get("logs", []))


def test_plan_summarize_then_rewrite(registry: BehaviorRegistry, interpreter: Interpreter, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_AUDIT", "1")
    ctx = {"data": {"text": "This is a long passage that needs a concise summary.", "max_words": 5, "tone": "professional"}}
    plan_path = Path("plans/summarize_then_rewrite.yaml")

    result = run_plan(str(plan_path), ctx, registry, interpreter)

    assert all(step["result"].get("ok") for step in result["steps"])
    behaviors = [step["behavior"] for step in result["steps"]]
    assert behaviors == ["summarize", "rewrite_style"]
    total_reward = result["total_reward"]
    step_rewards = sum(float(step["result"].get("reward") or 0.0) for step in result["steps"])
    assert total_reward == pytest.approx(step_rewards)


def test_mean_centered_bias_not_both_maxed() -> None:
    learner = BanditLearner(alpha=0.2)
    for _ in range(8):
        learner.update("rewrite_style", 1.0, "bucket")
        learner.update("summarize", 1.0, "bucket")

    biases = learner.adapter_biases("bucket")

    assert "rewrite_style" in biases and "summarize" in biases
    max_abs = max(abs(b) for b in biases.values())
    assert max_abs <= 0.3 + 1e-9
    assert not (
        biases["rewrite_style"] >= 0.29 and biases["summarize"] >= 0.29
    )


def test_better_behavior_gets_positive_bias() -> None:
    learner = BanditLearner(alpha=0.2)
    for _ in range(10):
        learner.update("summarize", 1.0, "default")
        learner.update("rewrite_style", 0.4, "default")

    biases = learner.adapter_biases("default")

    assert biases["summarize"] > 0
    assert biases["rewrite_style"] <= biases["summarize"]
    assert any(b < 0 for b in biases.values())


def test_contextual_biases_separate_buckets() -> None:
    learner = BanditLearner(alpha=0.2)
    for _ in range(12):
        learner.update("rewrite_style", 1.0, "style_hint")
    for _ in range(12):
        learner.update("summarize", 1.0, "long_text")

    style_biases = learner.adapter_biases("style_hint")
    long_biases = learner.adapter_biases("long_text")

    assert style_biases.get("rewrite_style", 0.0) > 0
    if "summarize" in style_biases:
        assert style_biases["summarize"] <= 0

    assert long_biases.get("summarize", 0.0) > 0
    if "rewrite_style" in long_biases:
        assert long_biases["rewrite_style"] <= 0


def test_feature_key_extraction() -> None:
    ctx_style = {"text": "Please rewrite this memo in a professional tone."}
    assert extract_feature_key(ctx_style) == "style_hint"

    compress_text = "This is a long paragraph that should be condensed to fewer words for clarity and easier reading."
    ctx_compress = {"text": compress_text, "data": {"text": compress_text, "max_words": 60}}
    assert extract_feature_key(ctx_compress) == "compress_hint"

    long_text = " ".join(["word"] * 120)
    ctx_long = {"data": {"text": long_text}}
    assert extract_feature_key(ctx_long) == "long_text"

    ctx_default = {"text": "Short prompt."}
    assert extract_feature_key(ctx_default) == "default"


def test_router_fallback_on_failed_behavior(registry: BehaviorRegistry, interpreter: Interpreter, capsys: pytest.CaptureFixture[str]) -> None:
    text = "This tone is impossible for the model to understand"
    ctx: Context = {"text": text, "data": {"text": text}}

    result = interpreter.execute("rewrite_style", ctx)
    assert result["reward"] == 0.0

    router = SimpleRouter(registry)
    chosen = router.choose(ctx)
    captured = capsys.readouterr().out

    assert chosen != "rewrite_style"
    assert "[router] fallback rewrite_style" in captured
    audit = ctx.get("router_audit")
    assert isinstance(audit, list) and audit[-1]["fallback_from"] == "rewrite_style"


def test_rewrite_style_success_reward(interpreter: Interpreter) -> None:
    text = "Please rewrite this memo in a professional tone for stakeholders."
    ctx: Context = {"text": text, "data": {"text": text}}

    result = interpreter.execute("rewrite_style", ctx)

    assert result["ok"] is True
    assert result["reward"] >= 1.0
    rewritten = ctx["data"]["rewritten_text"]
    assert "Please let me know" in rewritten


def test_failed_behavior_updates_context(interpreter: Interpreter) -> None:
    text = "This tone is impossible for the model to process"
    ctx: Context = {"text": text, "data": {"text": text}}

    result = interpreter.execute("rewrite_style", ctx)

    assert result["reward"] == 0.0
    router_state = ctx["router"]["recent_results"]["rewrite_style"]
    assert router_state["reward"] == 0.0
    assert ctx["data"]["rewards"]["rewrite_style"] == 0.0
    tone_checks = ctx["data"]["checks"]["rewrite_style"]
    assert tone_checks.get("tone_keyword_match") == 0.0
    assert "could not satisfy" in " ".join(result.get("logs", []))


def test_empty_input_handled(interpreter: Interpreter, registry: BehaviorRegistry, capsys: pytest.CaptureFixture[str]) -> None:
    ctx: Context = {"text": "", "data": {"text": ""}}

    result = interpreter.execute("rewrite_style", ctx)
    assert result["ok"] is True
    router = SimpleRouter(registry)
    chosen = router.choose(ctx)
    captured = capsys.readouterr().out

    assert chosen in {"rewrite_style", "summarize"}
    assert "[router] fallback" not in captured
