from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from core.bandit import UCB1


class WorkingMemory:
    """Lightweight scratch pad for inner loops."""

    def __init__(self, goal: str, context: Optional[Dict[str, Any]] = None) -> None:
        self.goal = goal
        self.context = dict(context or {})
        self.result_payload: Dict[str, Any] = {"goal": goal, "context": self.context}
        self._done = False

    def apply(self, partial: Optional[Dict[str, Any]]) -> None:
        """Merge partial updates into the working result."""
        if not partial:
            return
        self.result_payload.update(partial)
        if partial.get("final"):
            self._done = True

    def mark_done(self) -> None:
        self._done = True

    def done(self) -> bool:
        return self._done or bool(self.result_payload.get("final"))

    def snapshot(self) -> Dict[str, Any]:
        return dict(self.result_payload)


class EpisodicStore:
    """Linear record of inner steps."""

    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []

    def append(self, step: Dict[str, Any], out: Dict[str, Any], score: Dict[str, Any]) -> None:
        self._entries.append({"step": dict(step), "output": dict(out), "score": dict(score)})

    def __len__(self) -> int:  # pragma: no cover - simple proxy
        return len(self._entries)

    def __iter__(self) -> Iterable[Dict[str, Any]]:  # pragma: no cover
        return iter(self._entries)

    def entries(self) -> List[Dict[str, Any]]:
        return list(self._entries)


class ToolMemory:
    """Keeps lightweight performance statistics for affordance cues."""

    def __init__(self, initial: Optional[Dict[str, Any]] = None) -> None:
        base: Dict[str, Any] = {}
        if isinstance(initial, dict):
            base = {cue: dict(stats) for cue, stats in initial.items()}
        self._stats: Dict[str, Dict[str, Any]] = base
        self._bandits: Dict[str, UCB1] = {}
        for cue, stats in base.items():
            history = stats.get("history")
            bandit = self._bandits.setdefault(cue, UCB1())
            if isinstance(history, dict):
                for tool, values in history.items():
                    wins = int(values.get("win", 0) or 0)
                    total = int(values.get("total", 0) or 0)
                    for _ in range(total):
                        bandit.record(tool, 1.0 if wins > 0 else 0.0)
                        wins = max(0, wins - 1)

    def update_affordance(self, cue: str, tool: str, ok: bool) -> None:
        cue_stats = self._stats.setdefault(
            cue,
            {"tool": tool, "win": 0, "total": 0, "history": defaultdict(lambda: {"win": 0, "total": 0})},
        )
        history = cue_stats.setdefault("history", defaultdict(lambda: {"win": 0, "total": 0}))
        bucket = history[tool]
        bucket["total"] += 1
        if ok:
            bucket["win"] += 1

        bandit = self._bandits.setdefault(cue, UCB1())
        bandit.record(tool, 1.0 if ok else 0.0)

        best_tool, best_win, best_total = cue_stats["tool"], cue_stats["win"], cue_stats["total"]
        for tool_name, values in history.items():
            win = values["win"]
            total = values["total"]
            if total == 0:
                continue
            if total > best_total or (total == best_total and win > best_win):
                best_tool, best_win, best_total = tool_name, win, total

        cue_stats["tool"] = best_tool
        cue_stats["win"] = best_win
        cue_stats["total"] = best_total
        cue_stats["history"] = history

    def best_tool_for(self, cue: str) -> Optional[str]:
        bandit = self._bandits.get(cue)
        if bandit and bandit.has_arms():
            choice = bandit.pick()
            if choice:
                return choice

        stats = self._stats.get(cue)
        if not stats:
            return None
        return stats.get("tool")

    def should_fold(self, epi: Iterable[Any], every_n: int = 4) -> bool:
        length = len(epi) if hasattr(epi, "__len__") else sum(1 for _ in epi)
        return length > 0 and length % every_n == 0

    def fold(self, epi: Iterable[Dict[str, Any]]) -> List[Tuple[str, str, float]]:
        aggregates: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        for entry in epi:
            step = entry.get("step", {})
            cue = step.get("cue")
            tool = step.get("tool")
            score = entry.get("score", {})
            if not cue or not tool:
                continue
            win = 1.0 if score.get("ok") else 0.0
            aggregates[(cue, tool)].append(win)
        folded: List[Tuple[str, str, float]] = []
        for (cue, tool), wins in aggregates.items():
            winrate = sum(wins) / len(wins) if wins else 0.0
            folded.append((cue, tool, winrate))
        return folded

    def snapshot(self) -> Dict[str, Any]:
        output: Dict[str, Any] = {}
        for cue, stats in self._stats.items():
            history = stats.get("history")
            if isinstance(history, defaultdict):
                history_dict = {tool: dict(values) for tool, values in history.items()}
            elif isinstance(history, dict):
                history_dict = {tool: dict(values) for tool, values in history.items()}
            else:
                history_dict = {}
            output[cue] = {
                "tool": stats.get("tool"),
                "win": stats.get("win", 0),
                "total": stats.get("total", 0),
                "history": history_dict,
            }
        return output


__all__ = ["WorkingMemory", "EpisodicStore", "ToolMemory"]
