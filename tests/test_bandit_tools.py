from __future__ import annotations

from core.memory import ToolMemory


def test_bandit_prefers_successful_tool() -> None:
    tm = ToolMemory()
    cue = "compose"

    # Seed both tools with one observation each
    tm.update_affordance(cue, "good", True)
    tm.update_affordance(cue, "bad", False)

    picks = {"good": 1, "bad": 1}
    for idx in range(30):
        choice = tm.best_tool_for(cue)
        if not choice:
            choice = "good"
        picks[choice] = picks.get(choice, 0) + 1
        ok = choice == "good"
        tm.update_affordance(cue, choice, ok)

    total = sum(picks.values())
    assert picks["good"] / total >= 0.8
