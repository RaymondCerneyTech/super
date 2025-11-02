from __future__ import annotations

import pytest

from behaviors.code_edit import CodeEdit
from behaviors.meaning_infer import MeaningInfer
from core.interpreter import Interpreter
from core.planner import normalise_goal_flags, plan
from core.registry import BehaviorRegistry


def test_meaning_infer_detects_code_edit() -> None:
    behavior = MeaningInfer()
    ctx = {"text": "Please refactor the import section and update router logic."}

    result = behavior.run(ctx)  # type: ignore[arg-type]

    assert result["ok"] is True
    assert ctx["data"]["meaning"] == "code_edit"
    assert result["effects"] == ["meaning_inferred"]


def test_code_edit_behavior_handles_generic_request_with_placeholder() -> None:
    behavior = CodeEdit()
    ctx = {"text": "please make some improvements"}

    result = behavior.run(ctx)  # type: ignore[arg-type]

    assert result["ok"] is True
    assert "TODO: Apply code changes for" in result["output"]["code_update"]
    assert "code_update_detected" in result["effects"]


@pytest.mark.integration
def test_plan_handles_single_code_edit_step() -> None:
    registry = BehaviorRegistry().discover().load_meta()
    interpreter = Interpreter(registry)
    ctx = {
        "text": "refactor fetch_data to use async",
        "data": {
            "code": "def fetch_data():\n    return 42\n",
        },
    }
    goal_flags = normalise_goal_flags("formatted")

    plan_result = plan(
        goal_flags,
        ctx,
        registry,
        interpreter,
        cluster_bias="analytic",
        max_expansions=12,
    )

    steps = [name for name, _ in plan_result["steps"]]

    assert "meaning_infer" in steps
    assert "refactor_code" in steps
    assert "code_refactored" in plan_result["flags"]
    assert plan_result["goal_satisfied"] is True


@pytest.mark.integration
def test_plan_sequences_code_edit_pipeline() -> None:
    registry = BehaviorRegistry().discover().load_meta()
    interpreter = Interpreter(registry)
    original_code = (
        "def fetch_data():\n"
        "    return os.path.join('a', 'b')\n"
    )
    ctx = {
        "text": (
            "Refactor fetch_data to async, fix the imports, add a user login endpoint, "
            "and ensure the answer is formatted."
        ),
        "data": {
            "code": original_code,
        },
    }
    goal_flags = normalise_goal_flags("formatted")

    plan_result = plan(
        goal_flags,
        ctx,
        registry,
        interpreter,
        cluster_bias="analytic",
        max_expansions=18,
    )

    steps = [name for name, _ in plan_result["steps"]]

    assert "meaning_infer" in steps
    assert "refactor_code" in steps
    assert "code_edit" in steps
    assert "add_endpoint" in steps
    assert "document_formatting" in steps

    refactor_idx = steps.index("refactor_code")
    code_edit_idx = steps.index("code_edit")
    endpoint_idx = steps.index("add_endpoint")
    formatting_idx = steps.index("document_formatting")

    assert refactor_idx < code_edit_idx < endpoint_idx < formatting_idx
    assert {"code_refactored", "imports_fixed", "endpoint_generated", "formatted"}.issubset(
        set(plan_result["flags"])
    )
