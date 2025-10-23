from __future__ import annotations

from typing import Any, Dict, Iterable


def ensure_reward_dict(value: Any) -> Dict[str, float]:
    """
    Normalize an arbitrary reward value into a dictionary of float components.
    """
    reward: Dict[str, float] = {}
    if isinstance(value, dict):
        for key, component in value.items():
            try:
                reward[str(key)] = float(component)
            except (TypeError, ValueError):
                continue
    elif value is not None:
        try:
            reward["overall"] = float(value)
        except (TypeError, ValueError):
            reward = {}

    if reward and "overall" not in reward:
        reward["overall"] = aggregate_reward(reward)
    return reward


def aggregate_reward(reward: Dict[str, float]) -> float:
    """
    Produce a scalar estimate from a reward dictionary by averaging non-overall factors.
    """
    components = [float(v) for k, v in reward.items() if k != "overall"]
    if not components:
        components = [float(v) for v in reward.values()]
    return sum(components) / len(components) if components else 0.0


def merge_rewards(base: Dict[str, float], updates: Iterable[tuple[str, float]]) -> Dict[str, float]:
    """
    Update reward components with new values and refresh the aggregate.
    """
    for key, value in updates:
        try:
            base[key] = float(value)
        except (TypeError, ValueError):
            continue
    base["overall"] = aggregate_reward(base)
    return base


__all__ = ["ensure_reward_dict", "aggregate_reward", "merge_rewards"]
