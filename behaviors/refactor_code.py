from __future__ import annotations

import re
from typing import Dict, List

from core.interfaces import Behavior, Context, Result


FUNC_PATTERN = re.compile(
    r"^(?P<indent>\s*)def\s+(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*\((?P<params>[^)]*)\):",
    re.MULTILINE,
)


class RefactorCode(Behavior):
    name = "refactor_code"
    inputs: List[str] = ["text"]
    outputs: List[str] = ["code_update"]

    def run(self, ctx: Context) -> Result:
        data: Dict[str, object] = ctx.setdefault("data", {})
        text = ctx.get("text") or data.get("text") or ""
        source_code = data.get("code") or ""

        text_str = str(text)
        source_code_str = str(source_code)

        target_name = _extract_function_name(text_str)
        if not target_name or not source_code_str:
            message = "No function name or source code provided for refactor."
            return {
                "ok": False,
                "output": {},
                "effects": [],
                "logs": [message],
                "rationale": {"why": message, "evidence": [text_str[:120]]},
                "rewards": {"overall": 0.0},
            }

        updated_code, updated = _convert_function_to_async(source_code_str, target_name)
        if not updated:
            message = f"Could not locate function '{target_name}'."
            return {
                "ok": False,
                "output": {},
                "effects": [],
                "logs": [message],
                "rationale": {"why": message, "evidence": [text_str[:120]]},
                "rewards": {"overall": 0.0},
            }

        data["code_update"] = updated_code
        data["code"] = updated_code
        return {
            "ok": True,
            "output": {"code_update": updated_code},
            "effects": ["code_update_detected", "code_refactored"],
            "logs": [f"Refactored function '{target_name}' to async."],
            "rationale": {
                "why": f"Converted '{target_name}' to async function.",
                "evidence": [text_str[:200]],
            },
            "rewards": {"overall": 0.7},
        }


def _extract_function_name(task_text: str) -> str | None:
    match = re.search(r"refactor\s+([a-zA-Z_][a-zA-Z0-9_]*)", task_text.lower())
    if match:
        return match.group(1)
    return None


def _convert_function_to_async(source: str, target_name: str) -> tuple[str, bool]:
    def replacer(match: re.Match[str]) -> str:
        indent = match.group("indent")
        name = match.group("name")
        params = match.group("params")
        if name != target_name:
            return match.group(0)
        return f"{indent}async def {name}({params}):"

    updated_source, count = FUNC_PATTERN.subn(replacer, source)
    return updated_source, count > 0


__all__ = ["RefactorCode"]
