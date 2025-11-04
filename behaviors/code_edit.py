from __future__ import annotations

import datetime as _dt
import re
from typing import Any, Dict, Iterable, List, Tuple

from behaviors.add_endpoint import AddEndpoint
from core.interfaces import Behavior, Context, Result
from core.logs import append_jsonl


class CodeEdit(Behavior):
    name = "code_edit"
    inputs: List[str] = ["text"]
    outputs: List[str] = ["code_update"]

    def run(self, ctx: Context) -> Result:
        data: Dict[str, object] = ctx.setdefault("data", {})
        text = ctx.get("text") or data.get("text") or ""
        source_code = data.get("code") or ""
        text_str = str(text)

        lowered = text_str.lower()

        if "endpoint" in lowered and not source_code:
            endpoint_behavior = AddEndpoint()
            inner_ctx: Context = {"text": text_str, "data": data}
            result = endpoint_behavior.run(inner_ctx)
            _log_codegen_event(
                request=text_str,
                data=data,
                action="generate_endpoint",
                result=result,
                extra={"delegate": "add_endpoint"},
            )
            if result.get("ok", False):
                return result
            # fallback to placeholder if endpoint generation fails

        wants_import_fix = bool(
            re.search(r"\b(fix|clean|tidy|organize|update)(?:\s+\w+){0,3}\s+imports?\b", lowered)
        )
        if wants_import_fix:
            source = str(source_code)
            updated_code, info = _fix_imports(source)
            if not info["changed"]:
                message = (
                    "No import fixes were applied. Provide code containing missing or unused imports."
                )
                failure: Result = {
                    "ok": False,
                    "output": {},
                    "effects": [],
                    "logs": [message],
                    "rationale": {"why": message, "evidence": [text_str[:200]]},
                    "rewards": {"overall": 0.0},
                }
                _log_codegen_event(
                    request=text_str,
                    data=data,
                    action="fix_imports",
                    result=failure,
                    extra={"details": info},
                )
                return failure

            data["code_update"] = updated_code
            data["code"] = updated_code
            logs = []
            if info["added"]:
                logs.append(f"Added imports: {', '.join(sorted(info['added']))}")
            if info["removed"]:
                logs.append(f"Removed imports: {', '.join(sorted(info['removed']))}")
            if not logs:
                logs.append("Updated imports with no additions or removals.")
            success: Result = {
                "ok": True,
                "output": {"code_update": updated_code},
                "effects": ["code_update_detected", "imports_fixed"],
                "logs": logs,
                "rationale": {
                    "why": "Applied automated import fixes.",
                    "evidence": [text_str[:200]],
                },
                "rewards": {"overall": 0.65},
            }
            _log_codegen_event(
                request=text_str,
                data=data,
                action="fix_imports",
                result=success,
                extra={"details": info},
            )
            return success

        code_update = f"TODO: Apply code changes for: {text_str}"
        data["code_update"] = code_update

        fallback: Result = {
            "ok": True,
            "output": {"code_update": code_update},
            "effects": ["code_update_detected"],
            "logs": [f"Code edit requested: {text_str}"],
            "rationale": {
                "why": "Generated placeholder plan for code editing task.",
                "evidence": [text_str[:200]] if text_str else [],
            },
            "rewards": {"overall": 0.4},
        }
        _log_codegen_event(
            request=text_str,
            data=data,
            action="placeholder_plan",
            result=fallback,
        )
        return fallback


__all__ = ["CodeEdit"]

CODEGEN_LOG_PATH = "logs/codegen.jsonl"


COMMON_MODULES: Tuple[str, ...] = (
    "os",
    "sys",
    "json",
    "re",
    "pathlib",
    "requests",
    "pandas",
    "numpy",
    "datetime",
    "typing",
)

IMPORT_PATTERN = re.compile(r"^\s*(import|from)\s+(.+)")


def _fix_imports(source: str) -> Tuple[str, Dict[str, Iterable[str] | bool]]:
    lines = source.splitlines()
    if not lines:
        return source, {"added": [], "removed": [], "changed": False}

    import_indices: List[int] = []
    existing_modules: Dict[str, int] = {}

    for idx, line in enumerate(lines):
        stripped = line.strip()
        match = IMPORT_PATTERN.match(stripped)
        if not match:
            continue
        import_indices.append(idx)
        if stripped.startswith("import "):
            module_part = stripped[len("import ") :]
            module = module_part.split("as")[0].strip()
            module_root = module.split(".")[0]
            existing_modules[module_root] = idx
        elif stripped.startswith("from "):
            module_part = stripped[len("from ") :]
            module = module_part.split("import")[0].strip()
            module_root = module.split(".")[0]
            existing_modules[module_root] = idx

    body_without_imports = [
        line
        for idx, line in enumerate(lines)
        if idx not in import_indices
    ]
    body_text = "\n".join(body_without_imports)

    used_modules = {
        module
        for module in COMMON_MODULES
        if _module_is_used(module, body_text)
    }

    missing_modules = used_modules - set(existing_modules.keys())

    unused_modules = {
        module for module in existing_modules.keys() if module not in used_modules
    }

    changed = False
    updated_lines = lines[:]

    # Remove unused modules
    removed_modules: List[str] = []
    for module in unused_modules:
        idx = existing_modules[module]
        updated_lines[idx] = ""
        removed_modules.append(module)
        changed = True

    # Determine insertion point
    insert_position = 0
    if import_indices:
        insert_position = import_indices[-1] + 1
    else:
        for i, line in enumerate(updated_lines):
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                insert_position = i
                break

    added_modules: List[str] = []
    if missing_modules:
        import_block = [f"import {module}" for module in sorted(missing_modules)]
        updated_lines[insert_position:insert_position] = import_block + [""]
        added_modules.extend(sorted(missing_modules))
        changed = True

    updated_source = "\n".join(line for line in updated_lines if line.strip() or line == "")
    return updated_source, {"added": added_modules, "removed": removed_modules, "changed": changed}


def _module_is_used(module: str, code: str) -> bool:
    pattern = rf"\b{re.escape(module)}\s*(?:\.|\()"
    return re.search(pattern, code) is not None


def _log_codegen_event(
    *,
    request: str,
    data: Dict[str, object],
    action: str,
    result: Result,
    extra: Dict[str, Any] | None = None,
) -> None:
    record: Dict[str, Any] = {
        "ts": _now_iso(),
        "request": request,
        "action": action,
        "status": "ok" if result.get("ok", True) else "error",
        "effects": list(result.get("effects", [])) if result.get("effects") else [],
        "meaning": data.get("meaning"),
    }
    output = result.get("output", {})
    if isinstance(output, dict) and "code_update" in output:
        snippet = str(output["code_update"])
        record["code_preview"] = snippet[:2000]
        record["code_size"] = len(snippet)
    logs = result.get("logs")
    if isinstance(logs, list):
        record["logs"] = logs
    code_present = data.get("code")
    if code_present is not None:
        record["input_code_present"] = bool(str(code_present).strip())
    if extra:
        record["extra"] = extra
    try:
        append_jsonl(CODEGEN_LOG_PATH, record)
    except Exception:
        pass


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()
