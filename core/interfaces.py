# core/interfaces.py
from typing import Any, Dict, Iterable, List, Optional, Set, TypedDict


class Context(TypedDict, total=False):
    text: str
    data: Dict[str, Any]
    memory: Dict[str, Any]
    signals: Dict[str, Any]  # e.g., active app, file path, user prefs
    dry_run: bool


class Result(TypedDict, total=False):
    ok: bool
    output: Dict[str, Any]
    logs: List[str]
    checks: Dict[str, float]  # individual success metrics
    reward: float
    rewards: Dict[str, float]
    rationale: Dict[str, Any]
    effects: List[str]


class Behavior:
    name: str
    inputs: List[str]  # keys required in context.data
    outputs: List[str]  # keys written to context.data

    def run(self, ctx: Context) -> Result:
        raise NotImplementedError


class Router:
    def decide(self, ctx: Context) -> Dict[str, float]:
        """Return a dict of {behavior_name: probability}."""
        raise NotImplementedError


def evaluate_preconditions(preconditions: Iterable[str], ctx: Context, flags: Iterable[str]) -> bool:
    env = _build_environment(ctx, set(flags))
    allowed = {"len": len}
    for expr in preconditions:
        if not expr or expr.strip().lower() == "true":
            continue
        try:
            if not bool(eval(expr, {"__builtins__": {}}, {**allowed, **env})):
                return False
        except Exception:
            return False
    return True


def _build_environment(ctx: Context, flags: Set[str]) -> Dict[str, Any]:
    data = ctx.get("data", {}) if isinstance(ctx, dict) else {}
    text = ctx.get("text") or data.get("text") or ""
    summary = data.get("summary", "") if isinstance(data, dict) else ""
    policies = data.get("policies", []) if isinstance(data, dict) else []

    env: Dict[str, Any] = {
        "text_len": len(text.split()) if isinstance(text, str) else 0,
        "summary_len": len(summary.split()) if isinstance(summary, str) else 0,
        "has_text": bool(text.strip()) if isinstance(text, str) else False,
        "has_summary": bool(summary.strip()) if isinstance(summary, str) else False,
        "policies": policies,
        "flags": flags,
    }
    return env


__all__ = [
    "Context",
    "Result",
    "Behavior",
    "Router",
    "evaluate_preconditions",
]
