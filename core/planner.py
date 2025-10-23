from __future__ import annotations

import math
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core.learn import BanditLearner
from core.registry import BehaviorRegistry
from core.rewards import aggregate_reward, ensure_reward_dict

DEFAULT_BEHAVIORS = ["summarize", "rewrite_style"]
MONTE_CARLO_ROUNDS = 200
SAMPLING_STDDEV = 0.15


def plan_task(
    goal: str,
    registry: BehaviorRegistry,
    learner: Optional[BanditLearner] = None,
    history: Optional[Iterable[Dict[str, Any]]] = None,
    simulations: int = MONTE_CARLO_ROUNDS,
) -> List[str]:
    """
    Build a goal-oriented plan by simulating candidate behavior sequences and returning the best.
    """
    available = set(registry.list())
    sequences = _candidate_sequences(goal, available)
    if not sequences:
        return []

    behavior_stats = _collect_behavior_stats(learner, history)
    rng = random.Random(42)

    evaluated: List[Tuple[float, List[str]]] = []
    for sequence in sequences:
        expected_value = _simulate_sequence(sequence, behavior_stats, simulations, rng)
        evaluated.append((expected_value, sequence))

    evaluated.sort(key=lambda item: item[0], reverse=True)
    return evaluated[0][1]


def _candidate_sequences(goal: str, available: set[str]) -> List[List[str]]:
    goal_lower = goal.lower()
    sequences: List[List[str]] = []

    wants_rewrite = any(keyword in goal_lower for keyword in ("rewrite", "tone", "professional", "casual"))
    wants_summary = any(
        keyword in goal_lower for keyword in ("summarize", "summary", "condense", "shorten", "tl;dr", "compress")
    )

    def add_sequence(seq: Sequence[str]) -> None:
        if all(step in available for step in seq) and list(seq) not in sequences:
            sequences.append(list(seq))

    if wants_summary:
        add_sequence(["summarize"])
        add_sequence(["summarize", "rewrite_style"])

    if wants_rewrite:
        add_sequence(["rewrite_style"])

    if not sequences:
        for behavior in DEFAULT_BEHAVIORS:
            if behavior in available:
                add_sequence([behavior])
        if {"summarize", "rewrite_style"}.issubset(available):
            add_sequence(["summarize", "rewrite_style"])

    return sequences


def _collect_behavior_stats(
    learner: Optional[BanditLearner],
    history: Optional[Iterable[Dict[str, Any]]],
) -> Dict[str, float]:
    stats: Dict[str, List[float]] = {}

    if learner is not None:
        for feature_map in learner.values.values():
            for behavior, mean_reward in feature_map.items():
                stats.setdefault(behavior, []).append(float(mean_reward))

    if history:
        for entry in history:
            behavior = entry.get("behavior")
            if not behavior:
                continue
            reward = ensure_reward_dict(entry.get("reward"))
            stats.setdefault(behavior, []).append(aggregate_reward(reward))

    aggregated: Dict[str, float] = {}
    for behavior, values in stats.items():
        if values:
            aggregated[behavior] = sum(values) / len(values)

    return aggregated


def _simulate_sequence(
    sequence: Sequence[str],
    behavior_stats: Dict[str, float],
    simulations: int,
    rng: random.Random,
) -> float:
    if simulations <= 0:
        simulations = MONTE_CARLO_ROUNDS

    total = 0.0
    for _ in range(simulations):
        seq_value = 0.0
        for behavior in sequence:
            mean = behavior_stats.get(behavior, 0.55)
            sample = rng.gauss(mean, SAMPLING_STDDEV)
            sample = min(max(sample, 0.0), 1.0)
            seq_value += sample
        total += seq_value

    return total / simulations


__all__ = ["plan_task"]
