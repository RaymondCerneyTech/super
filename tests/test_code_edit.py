from __future__ import annotations

import textwrap

from behaviors.code_edit import CodeEdit
from behaviors.refactor_code import RefactorCode


def test_refactor_simple_function_to_async() -> None:
    behavior = RefactorCode()
    original = textwrap.dedent(
        """
        def fetch_data():
            return 42
        """
    ).strip()

    ctx = {
        "text": "Please refactor fetch_data to be async",
        "data": {"code": original},
    }

    result = behavior.run(ctx)  # type: ignore[arg-type]

    assert result["ok"] is True
    assert "code_refactored" in result["effects"]
    updated = result["output"]["code_update"]
    assert "async def fetch_data()" in updated


def test_fix_imports_adds_missing_modules() -> None:
    behavior = CodeEdit()
    original = textwrap.dedent(
        """
        def create_temp_dir():
            return os.path.join(\"/tmp\", \"example\")
        """
    ).strip()

    ctx = {
        "text": "fix imports in this snippet",
        "data": {"code": original},
    }

    result = behavior.run(ctx)  # type: ignore[arg-type]

    assert result["ok"] is True
    assert "imports_fixed" in result["effects"]
    updated = result["output"]["code_update"]
    assert "import os" in updated
    assert "create_temp_dir" in updated


def test_fix_imports_removes_unused_modules() -> None:
    behavior = CodeEdit()
    original = textwrap.dedent(
        """
        import json

        def answer():
            return 42
        """
    ).strip()

    ctx = {
        "text": "Please fix imports",
        "data": {"code": original},
    }

    result = behavior.run(ctx)  # type: ignore[arg-type]

    assert result["ok"] is True
    assert "imports_fixed" in result["effects"]
    updated = result["output"]["code_update"]
    assert "import json" not in updated


def test_generate_fastapi_endpoint() -> None:
    behavior = CodeEdit()
    ctx = {
        "text": "add an API endpoint for user login",
        "data": {},
    }

    result = behavior.run(ctx)  # type: ignore[arg-type]

    assert result["ok"] is True
    assert "endpoint_generated" in result["effects"]
    code = result["output"]["code_update"]
    assert "@router.post" in code
    assert "async def login_user" in code
    assert "BaseModel" in code
