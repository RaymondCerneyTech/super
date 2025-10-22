# core/interpreter.py
from typing import Dict, Any
from core.interfaces import Context, Result, Behavior

class Interpreter:
    def __init__(self, registry: Dict[str, Behavior]):
        self.registry = registry

    def execute(self, name: str, ctx: Context) -> Result:
        bh = self.registry[name]
        # preconditions
        for k in bh.inputs:
            if k not in ctx and k not in ctx.get("data", {}):
                return {"ok": False, "logs": [f"Missing input: {k}"], "reward": 0.0}
        res = bh.run(ctx)
        # write outputs back into ctx.data
        for k, v in (res.get("output") or {}).items():
            ctx.setdefault("data", {})[k] = v
        return res
