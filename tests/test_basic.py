from pathlib import Path

import pytest

from core.features import extract_feature_key
from core.interpreter import Interpreter
from core.learn import BanditLearner
from core.planner import plan_task
from core.plans import run_plan
from core.registry import BehaviorRegistry
from core.rewards import aggregate_reward, ensure_reward_dict
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

    reward = ensure_reward_dict(result["reward"])
    assert reward["length_leq"] == 1.0
    assert "length_leq" in result["checks"]
    assert result["checks"]["length_leq"] == 1.0


def test_router_prefers_summarize_for_long_text(registry: BehaviorRegistry) -> None:
    router = SimpleRouter(registry)
    long_text = " ".join(["This summary aims to condense important points."] * 10)
    ctx = {"text": long_text, "data": {"text": long_text, "max_words": 60}}

    choice = router.choose(ctx)
    assert choice == "summarize"


def test_router_prefers_rewrite_style_for_professional_tone(registry: BehaviorRegistry) -> None:
    router = SimpleRouter(registry)
    text = "Please rewrite this message in a professional tone with a polished style."
    ctx = {"text": text, "data": {"text": text, "max_words": 80}}

    choice = router.choose(ctx)
    assert choice == "rewrite_style"


def test_router_picks_rewrite_style_for_professional_tone(registry: BehaviorRegistry) -> None:
    router = SimpleRouter(registry)
    ctx = {"data": {"text": "Please rewrite this memo in a professional tone", "max_words": 80}}

    choice = router.choose(ctx)
    assert choice == "rewrite_style"


def test_router_prefers_grammar_correction(registry: BehaviorRegistry) -> None:
    router = SimpleRouter(registry)
    text = "Please correct the grammar and clean up the prose in this paragraph."
    ctx = {"text": text, "data": {"text": text, "max_words": 80, "grammar_mode": "aggressive"}}

    choice = router.choose(ctx)
    assert choice == "grammar_correction"


def test_router_prefers_sentiment_analysis(registry: BehaviorRegistry) -> None:
    router = SimpleRouter(registry)
    text = "Please run sentiment analysis and tell me if customers feel angry, joyful, or sarcastic about our latest update."
    ctx = {"text": text, "data": {"text": text, "max_words": 80}}

    choice = router.choose(ctx)
    assert choice == "sentiment_analysis"


def test_bandit_bias_moves_toward_success() -> None:
    learner = BanditLearner()
    for _ in range(10):
        learner.update("summarize", {"overall": 1.0}, "default")

    biases = learner.adapter_biases("default")
    assert biases["summarize"] >= 0


def test_rewrite_style_invalid_tone(interpreter: Interpreter) -> None:
    ctx = {"data": {"text": "Please adjust this message.", "tone": "silly"}}

    result = interpreter.execute("rewrite_style", ctx)

    assert result["ok"] is False
    assert ensure_reward_dict(result["reward"])["overall"] == 0.0
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
    step_rewards = sum(aggregate_reward(ensure_reward_dict(step["reward"])) for step in result["steps"])
    assert total_reward == pytest.approx(step_rewards)


def test_mean_centered_bias_not_both_maxed() -> None:
    learner = BanditLearner(alpha=0.2)
    for _ in range(8):
        learner.update("rewrite_style", {"overall": 1.0}, "bucket")
        learner.update("summarize", {"overall": 1.0}, "bucket")

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
        learner.update("summarize", {"overall": 1.0, "clarity": 1.0}, "default")
        learner.update("rewrite_style", {"overall": 0.4, "tone_accuracy": 0.4}, "default")

    biases = learner.adapter_biases("default")

    assert biases["summarize"] > 0
    assert biases["rewrite_style"] <= biases["summarize"]
    assert any(b < 0 for b in biases.values())


def test_contextual_biases_separate_buckets() -> None:
    learner = BanditLearner(alpha=0.2)
    for _ in range(12):
        learner.update("rewrite_style", {"overall": 1.0, "tone_accuracy": 1.0}, "style_hint")
    for _ in range(12):
        learner.update("summarize", {"overall": 1.0, "length": 1.0}, "long_text")

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


def test_plan_task_prefers_high_reward_sequence(registry: BehaviorRegistry) -> None:
    learner = BanditLearner(alpha=0.3)
    for _ in range(5):
        learner.update("summarize", {"overall": 0.9, "length": 0.95}, "default")
        learner.update("rewrite_style", {"overall": 0.2, "tone_accuracy": 0.25}, "default")

    sequence = plan_task(
        "Please summarize this report and rewrite it professionally.",
        registry,
        learner,
    )

    assert sequence == ["summarize", "rewrite_style"]


def test_plan_task_uses_history_when_no_learner_data(registry: BehaviorRegistry) -> None:
    history = [
        {"behavior": "summarize", "reward": {"overall": 0.8, "length": 0.85}},
        {"behavior": "rewrite_style", "reward": {"overall": 0.15, "tone_accuracy": 0.2}},
    ]

    sequence = plan_task(
        "Rewrite this text after you condense it.",
        registry,
        None,
        history=history,
        simulations=50,
    )

    assert sequence == ["summarize", "rewrite_style"]


def test_run_plan_records_history(registry: BehaviorRegistry, interpreter: Interpreter, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_AUDIT", "1")
    ctx: Context = {"data": {"text": "Plan history capture example.", "max_words": 10, "tone": "professional"}}
    plan_path = Path("plans/summarize_then_rewrite.yaml")

    result = run_plan(str(plan_path), ctx, registry, interpreter)

    assert len(ctx.get("plan_history", [])) == len(result["steps"])
    assert ctx["plan_history"][0]["plan"] == "summarize_then_rewrite"
    assert isinstance(ctx["plan_history"][0]["reward"], dict)
    assert "overall" in ctx["plan_history"][0]["reward"]


def test_grammar_correction_behavior(interpreter: Interpreter) -> None:
    ctx: Context = {"data": {"text": "this is a test. i hope it works.", "max_words": 60}}
    result = interpreter.execute("grammar_correction", ctx)

    corrected = ctx["data"]["corrected_text"]
    assert corrected.startswith("This")
    assert "I hope" in corrected
    reward = ensure_reward_dict(result["reward"])
    assert "relevance" in reward


def test_grammar_correction_aggressive_mode(interpreter: Interpreter) -> None:
    ctx: Context = {
        "data": {
            "text": "gonna go now, see ya later!! this needs cleanup",
            "max_words": 60,
            "grammar_mode": "aggressive",
        }
    }
    result = interpreter.execute("grammar_correction", ctx)
    corrected = ctx["data"]["corrected_text"]
    assert "going to go now" in corrected.lower()
    assert corrected.endswith(".")
    assert "Aggressive grammar refinement enabled." in result["logs"]


def test_sentiment_analysis_behavior(interpreter: Interpreter) -> None:
    ctx: Context = {"data": {"text": "I love this product, it is amazing!", "max_words": 80}}
    result = interpreter.execute("sentiment_analysis", ctx)

    sentiment = ctx["data"]["sentiment"]
    score = ctx["data"]["sentiment_score"]
    assert sentiment == "positive"
    assert 0.5 <= score <= 1.0
    assert ctx["data"]["sentiment_category"] == "joy"
    assert ctx["data"]["analysis_text"]
    reward = ensure_reward_dict(result["reward"])
    assert "overall" in reward


def test_sentiment_analysis_sarcasm(interpreter: Interpreter) -> None:
    ctx: Context = {"data": {"text": "Great, another delay in the release, yeah right.", "max_words": 80}}
    result = interpreter.execute("sentiment_analysis", ctx)
    assert ctx["data"]["sentiment_category"] == "sarcasm"
    assert ctx["data"]["sentiment"] == "negative"
    assert "sarcasm" in result["logs"][0].lower()


def test_router_fallback_on_failed_behavior(registry: BehaviorRegistry, interpreter: Interpreter, capsys: pytest.CaptureFixture[str]) -> None:
    text = "This tone is impossible for the model to understand"
    ctx: Context = {"text": text, "data": {"text": text, "max_words": 80}}

    result = interpreter.execute("rewrite_style", ctx)
    assert ensure_reward_dict(result["reward"])["overall"] == 0.0

    router = SimpleRouter(registry)
    chosen = router.choose(ctx)
    captured = capsys.readouterr().out

    assert chosen != "rewrite_style"
    assert "[router] fallback rewrite_style" in captured
    audit = ctx.get("router_audit")
    assert isinstance(audit, list) and audit[-1]["fallback_from"] == "rewrite_style"


def test_rewrite_style_success_reward(interpreter: Interpreter) -> None:
    text = "Please rewrite this memo in a professional tone for stakeholders."
    ctx: Context = {"text": text, "data": {"text": text, "max_words": 80}}

    result = interpreter.execute("rewrite_style", ctx)

    assert result["ok"] is True
    reward = ensure_reward_dict(result["reward"])
    assert reward["tone_keyword_match"] >= 1.0
    assert reward["overall"] >= 1.0
    rewritten = ctx["data"]["rewritten_text"]
    assert "Please let me know" in rewritten


def test_failed_behavior_updates_context(interpreter: Interpreter) -> None:
    text = "This tone is impossible for the model to process"
    ctx: Context = {"text": text, "data": {"text": text, "max_words": 80}}

    result = interpreter.execute("rewrite_style", ctx)

    reward = ensure_reward_dict(result["reward"])
    assert reward["overall"] == 0.0
    router_state = ctx["router"]["recent_results"]["rewrite_style"]
    assert ensure_reward_dict(router_state["reward"])["overall"] == 0.0
    assert ensure_reward_dict(ctx["data"]["rewards"]["rewrite_style"])["overall"] == 0.0
    tone_checks = ctx["data"]["checks"]["rewrite_style"]
    assert tone_checks.get("tone_keyword_match") == 0.0
    assert "could not satisfy" in " ".join(result.get("logs", []))


def test_empty_input_handled(interpreter: Interpreter, registry: BehaviorRegistry, capsys: pytest.CaptureFixture[str]) -> None:
    ctx: Context = {"text": "", "data": {"text": "", "max_words": 40}}

    result = interpreter.execute("rewrite_style", ctx)
    assert result["ok"] is True
    router = SimpleRouter(registry)
    chosen = router.choose(ctx)
    captured = capsys.readouterr().out

    assert chosen in {"rewrite_style", "summarize", "grammar_correction"}
    assert "[router] fallback" not in captured


def test_multidimensional_reward_structure(interpreter: Interpreter) -> None:
    text = "Please rewrite this memo in a professional tone while keeping it concise."
    ctx: Context = {"text": text, "data": {"text": text}}

    result = interpreter.execute("rewrite_style", ctx)
    reward = ensure_reward_dict(result["reward"])

    assert "tone_keyword_match" in reward
    assert "overall" in reward
    assert reward["tone_keyword_match"] <= 1.0


def test_delayed_reward_adjustment(interpreter: Interpreter) -> None:
    failing_text = "This tone is impossible for the model to deliver"
    ctx: Context = {"text": failing_text, "data": {"text": failing_text}}

    failure = interpreter.execute("rewrite_style", ctx)
    failure_reward = ensure_reward_dict(failure["reward"])
    assert failure_reward["tone_keyword_match"] == 0.0

    success_text = "Please rewrite this memo in a professional tone."
    ctx["text"] = success_text
    ctx["data"]["text"] = success_text

    success = interpreter.execute("rewrite_style", ctx)
    success_reward = ensure_reward_dict(success["reward"])
    assert success_reward["tone_keyword_match"] >= 1.0

    adjusted = ensure_reward_dict(ctx["data"]["rewards"]["rewrite_style"])
    assert adjusted["tone_keyword_match"] == pytest.approx(0.5)
    backlog = ctx.get("reward_backlog", {})
    assert not backlog.get("tone_keyword_match")
