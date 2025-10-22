# behaviors/summarize.py
from core.interfaces import Behavior, Context, Result

class Summarize(Behavior):
    name = "summarize"
    inputs = ["text"]
    outputs = ["summary"]

    def run(self, ctx: Context) -> Result:
        text = ctx.get("text") or ctx["data"]["text"]
        N = int(ctx["data"].get("max_words", 120))
        # naive summarizer: first N words (placeholder)
        words = text.split()
        summary = " ".join(words[:N])
        ok = len(summary.split()) <= N
        return {
            "ok": ok,
            "output": {"summary": summary},
            "logs": [f"Summarized to <= {N} words"],
            "checks": {"length_leq": 1.0 if ok else 0.0},
            "reward": 1.0 if ok else 0.0,
        }
