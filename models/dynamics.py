from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


if torch is not None and nn is not None:

    class _DynamicsMLP(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 32) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.net(x)


    class DynamicsEnsemble(nn.Module):
        def __init__(
            self,
            state_dim: int,
            action_dim: int,
            ensemble_size: int = 5,
            hidden_dim: int = 32,
            device: str | torch.device = "cpu",
        ) -> None:
            super().__init__()
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.ensemble_size = ensemble_size
            self.models = nn.ModuleList(
                [
                    _DynamicsMLP(state_dim + action_dim, state_dim, hidden_dim)
                    for _ in range(ensemble_size)
                ]
            )
            self.to(device)

        def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            inputs = torch.cat([s, a], dim=-1)
            outputs = []
            for model in self.models:
                outputs.append(model(inputs))
            return torch.stack(outputs, dim=0)

        def predict(self, state_vec: Sequence[float], action_vec: Sequence[float]) -> Tuple[torch.Tensor, torch.Tensor]:
            self.eval()
            with torch.no_grad():
                s = torch.tensor(state_vec, dtype=torch.float32)
                a = torch.tensor(action_vec, dtype=torch.float32)
                inputs = torch.cat([s, a], dim=-1)
                outputs = []
                for model in self.models:
                    delta = model(inputs)
                    outputs.append(s + delta)
                predictions = torch.stack(outputs, dim=0)
                mean_next = predictions.mean(dim=0)
                variance = predictions.var(dim=0).mean()
                return mean_next, variance


    @dataclass
    class DynamicsTrainingState:
        ensemble: DynamicsEnsemble
        optimizers: List[torch.optim.Optimizer]


    def build_ensemble(
        state_dim: int,
        action_dim: int,
        ensemble_size: int = 5,
        hidden_dim: int = 32,
        device: str | torch.device = "cpu",
    ) -> DynamicsTrainingState:
        ensemble = DynamicsEnsemble(state_dim, action_dim, ensemble_size=ensemble_size, hidden_dim=hidden_dim, device=device)
        optimizers = [
            torch.optim.Adam(model.parameters(), lr=1e-3)
            for model in ensemble.models
        ]
        return DynamicsTrainingState(ensemble=ensemble, optimizers=optimizers)


    def train_step(state: DynamicsTrainingState, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> float:
        S, A, SP = batch
        total_loss = 0.0
        for model, optim in zip(state.ensemble.models, state.optimizers):
            optim.zero_grad()
            inputs = torch.cat([S, A], dim=-1)
            pred_delta = model(inputs)
            loss = F.mse_loss(S + pred_delta, SP)
            loss.backward()
            optim.step()
            total_loss += loss.item()
        return total_loss / len(state.ensemble.models)


    def save_ensemble(state: DynamicsTrainingState, path: str) -> None:
        torch.save(
            {
                "model_state": state.ensemble.state_dict(),
                "state_dim": state.ensemble.state_dim,
                "action_dim": state.ensemble.action_dim,
                "ensemble_size": state.ensemble.ensemble_size,
            },
            path,
        )


    def load_ensemble(path: str, device: str | torch.device = "cpu") -> DynamicsEnsemble:
        checkpoint = torch.load(path, map_location=device)
        state_dim = int(checkpoint["state_dim"])
        action_dim = int(checkpoint["action_dim"])
        ensemble_size = int(checkpoint.get("ensemble_size", 5))
        ensemble = DynamicsEnsemble(state_dim, action_dim, ensemble_size=ensemble_size, device=device)
        ensemble.load_state_dict(checkpoint["model_state"])
        ensemble.eval()
        return ensemble


else:  # torch not available

    class DynamicsEnsemble:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("PyTorch is required for dynamics modelling")

        def predict(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for dynamics modelling")


    @dataclass
    class DynamicsTrainingState:
        ensemble: None
        optimizers: List[None]


    def build_ensemble(*args, **kwargs) -> DynamicsTrainingState:  # type: ignore[override]
        raise RuntimeError("PyTorch is required for dynamics modelling")


    def train_step(*args, **kwargs) -> float:  # type: ignore[override]
        raise RuntimeError("PyTorch is required for dynamics modelling")


    def save_ensemble(*args, **kwargs) -> None:  # type: ignore[override]
        raise RuntimeError("PyTorch is required for dynamics modelling")


    def load_ensemble(*args, **kwargs) -> DynamicsEnsemble:  # type: ignore[override]
        raise RuntimeError("PyTorch is required for dynamics modelling")


__all__ = [
    "DynamicsEnsemble",
    "DynamicsTrainingState",
    "build_ensemble",
    "train_step",
    "save_ensemble",
    "load_ensemble",
]
