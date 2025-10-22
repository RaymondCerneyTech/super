# core/router.py
from typing import Dict
from core.interfaces import Context

class SimpleRouter:
    def __init__(self, behaviors: Dict[str, Dict], adapters: Dict[str, Dict]):
        self.behaviors = behaviors  # meta specs
        self.adapters = adapters    # small biases, e.g., {"summarize": +0.5}

    def decide(self, ctx: Context) -> Dict[str, float]:
        text = (ctx.get("text") or ctx["data"].get("text") or "").lower()
        logits = {name: 0.1 for name in self.behaviors.keys()}
        # keyword heuristic
        for name, meta in self.behaviors.items():
            kws = meta.get("keywords", [])
            logits[name] += sum(text.count(k) for k in kws)
        # adapter bias (optional)
        for name, b in self.adapters.items():
            logits[name] = logits.get(name, 0.1) + b
        # softmax
        mx = max(logits.values())
        exps = {k: pow(2.71828, v - mx) for k,v in logits.items()}
        Z = sum(exps.values()) or 1.0
        return {k: exps[k]/Z for k in logits}
