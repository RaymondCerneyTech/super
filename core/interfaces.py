# core/interfaces.py
from typing import Any, Dict, List, TypedDict, Optional

class Context(TypedDict, total=False):
    text: str
    data: Dict[str, Any]
    memory: Dict[str, Any]
    signals: Dict[str, Any]   # e.g., active app, file path, user prefs
    dry_run: bool

class Result(TypedDict, total=False):
    ok: bool
    output: Dict[str, Any]
    logs: List[str]
    checks: Dict[str, float]  # individual success metrics
    reward: float

class Behavior:
    name: str
    inputs: List[str]    # keys required in context.data
    outputs: List[str]   # keys written to context.data

    def run(self, ctx: Context) -> Result:
        raise NotImplementedError

class Router:
    def decide(self, ctx: Context) -> Dict[str, float]:
        """Return a dict of {behavior_name: probability}."""
        raise NotImplementedError
