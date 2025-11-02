from __future__ import annotations

from typing import Dict, List

from core.interfaces import Behavior, Context, Result


class AddEndpoint(Behavior):
    name = "add_endpoint"
    inputs: List[str] = ["text"]
    outputs: List[str] = ["code_update"]

    def run(self, ctx: Context) -> Result:
        data: Dict[str, object] = ctx.setdefault("data", {})
        text = ctx.get("text") or data.get("text") or ""
        text_str = str(text)

        endpoint_name = _infer_endpoint_name(text_str)
        method = _infer_http_method(text_str)

        code_snippet = _generate_fastapi_endpoint(endpoint_name, method)
        data["code_update"] = code_snippet
        data["code"] = code_snippet

        logs = [f"Generated FastAPI {method} endpoint '{endpoint_name}'."]
        return {
            "ok": True,
            "output": {"code_update": code_snippet},
            "effects": ["code_update_detected", "endpoint_generated"],
            "logs": logs,
            "rationale": {
                "why": f"Created FastAPI endpoint scaffold for '{endpoint_name}'.",
                "evidence": [text_str[:200]],
            },
            "rewards": {"overall": 0.7},
        }


def _infer_endpoint_name(task_text: str) -> str:
    lowered = task_text.lower()
    if "login" in lowered:
        return "login_user"
    if "signup" in lowered or "register" in lowered:
        return "register_user"
    if "logout" in lowered:
        return "logout_user"
    if "profile" in lowered:
        return "get_user_profile"
    return "new_endpoint"


def _infer_http_method(task_text: str) -> str:
    lowered = task_text.lower()
    if "create" in lowered or "add" in lowered or "register" in lowered or "login" in lowered:
        return "POST"
    if "update" in lowered or "put" in lowered:
        return "PUT"
    if "delete" in lowered or "remove" in lowered:
        return "DELETE"
    return "GET"


def _generate_fastapi_endpoint(name: str, method: str) -> str:
    http_decorator = {
        "GET": "@router.get",
        "POST": "@router.post",
        "PUT": "@router.put",
        "DELETE": "@router.delete",
    }.get(method, "@router.get")

    request_model = f"{name.title().replace('_', '')}Request"
    response_model = f"{name.title().replace('_', '')}Response"

    body_params = ""
    if method in {"POST", "PUT", "DELETE"}:
        body_params = f"data: {request_model}"

    code = f'''from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

router = APIRouter()


class {request_model}(BaseModel):
    username: str
    password: str


class {response_model}(BaseModel):
    success: bool
    message: str


{http_decorator}(
    "/{name.replace("_", "-")}",
    response_model={response_model},
    summary="{method.title()} {name.replace("_", " ")}",
)
async def {name}({body_params}):
    """
    Auto-generated endpoint stub.
    TODO: Implement business logic for {name}.
    """
    return {response_model}(success=True, message="{name} executed")
'''
    return code


__all__ = ["AddEndpoint"]
