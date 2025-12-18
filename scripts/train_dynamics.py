from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch

from core.world_model import DEFAULT_ACTION_SPACE, action_to_vector, state_to_vector
from models.dynamics import build_ensemble, save_ensemble, train_step


def _read_rollouts(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    records: List[Dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _prepare_batches(records: Iterable[Dict[str, object]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    states: List[List[float]] = []
    actions: List[List[float]] = []
    next_states: List[List[float]] = []

    for record in records:
        state = record.get("s") or {}
        action = record.get("a") or {}
        next_state = record.get("s_prime") or {}
        if not isinstance(state, dict) or not isinstance(action, dict) or not isinstance(next_state, dict):
            continue
        action_name = str(action.get("name") or "")
        states.append(state_to_vector(state))
        actions.append(action_to_vector(action_name))
        next_states.append(state_to_vector(next_state))

    if not states:
        raise RuntimeError("No usable rollouts found. Generate data by running plans with log_rollouts.")

    S = torch.tensor(states, dtype=torch.float32)
    A = torch.tensor(actions, dtype=torch.float32)
    SP = torch.tensor(next_states, dtype=torch.float32)
    return S, A, SP


def train_model(
    rollouts_path: Path,
    output_path: Path,
    *,
    steps: int = 500,
    batch_size: int = 32,
    seed: int = 0,
) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    records = _read_rollouts(rollouts_path)
    S, A, SP = _prepare_batches(records)

    dataset_size = S.size(0)
    state = build_ensemble(state_dim=S.size(1), action_dim=A.size(1), ensemble_size=5, hidden_dim=32)
    print(f"Training ensemble on {dataset_size} transitions for {steps} steps")

    indices = list(range(dataset_size))
    for step in range(1, steps + 1):
        random.shuffle(indices)
        batch_idx = indices[: min(batch_size, dataset_size)]
        loss = train_step(
            state,
            (
                S[batch_idx],
                A[batch_idx],
                SP[batch_idx],
            ),
        )
        if step % 50 == 0 or step == 1 or step == steps:
            print(f"[train_dynamics] step={step} loss={loss:.6f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_ensemble(state, str(output_path))
    print(f"Saved dynamics ensemble to {output_path}")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train small dynamics ensemble from rollout logs.")
    parser.add_argument("--rollouts", default=".ai/rollouts.jsonl", help="Path to rollout JSONL file")
    parser.add_argument("--output", default=".ai/dynamics_ensemble.pt", help="Destination path for trained weights")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    rollouts_path = Path(args.rollouts)
    output_path = Path(args.output)
    if not rollouts_path.exists():
        raise SystemExit(f"No rollout data found at {rollouts_path}. Run plans with log_rollouts first.")
    train_model(rollouts_path, output_path, steps=args.steps, batch_size=args.batch_size, seed=args.seed)


if __name__ == "__main__":  # pragma: no cover
    main()
