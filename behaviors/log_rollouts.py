from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from core.interfaces import Behavior, Context, Result
from core.logs import append_jsonl
from core.rewards import citation_coverage, faithfulness, verbosity_score
from core.world_model import compact_state


ROLL_OUT_PATH = Path(".ai") / "rollouts.jsonl"


def _compute_reward(data: Dict[str, Any]) -> float:
    llama_output = data.get("llama_output")
    if isinstance(llama_output, dict):
        text = str(llama_output.get("text") or "")
    else:
        text = str(llama_output or "")
    if not text.strip():
        return 0.0

    passages = data.get("passages") or []
    context = " ".join(str(entry.get("text", "")) for entry in passages[:3]) if isinstance(passages, list) else ""
    coverage = citation_coverage(text)
    faithful = faithfulness(text, context) if context else 0.0
    target_words = int(data.get("min_words") or 600)
    verbosity = verbosity_score(text, max(target_words, 1))
    return float((coverage + faithful + verbosity) / 3.0)


class LogRollouts(Behavior):
    name = "log_rollouts"
    inputs: List[str] = []
    outputs: List[str] = []

    def run(self, ctx: Context) -> Result:
        data = ctx.setdefault("data", {})
        prev_state = data.get("_prev_state")
        last_action = data.get("_last_action")
        if not isinstance(prev_state, dict) or not isinstance(last_action, dict):
            return {
                "ok": True,
                "logs": ["log_rollouts: skipping (missing state/action)"],
                "checks": {},
                "reward": 0.0,
                "effects": [],
            }

        current_state = compact_state(data)
        reward = _compute_reward(data)
        step_index = int(data.get("_trajectory_step") or 0)

        record = {
            "t": step_index,
            "s": prev_state,
            "a": last_action,
            "s_prime": current_state,
            "r": reward,
        }
        append_jsonl(ROLL_OUT_PATH, record)

        data["_prev_state"] = current_state
        data["_last_action"] = None
        data["_trajectory_step"] = step_index + 1

        return {
            "ok": True,
            "logs": ["log_rollouts: recorded transition"],
            "checks": {},
            "reward": reward,
            "effects": ["rollout_logged"],
        }


__all__ = ["LogRollouts"]
