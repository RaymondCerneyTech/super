from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from core.interfaces import Behavior, Context, Result
from core.world_model import DEFAULT_ACTION_SPACE, action_to_vector, candidate_actions, compact_state, state_to_vector
from models.dynamics import load_ensemble


DEFAULT_MODEL_PATH = Path(".ai") / "world_model" / "dynamics_ensemble.pt"


class MPCPlan(Behavior):
    name = "mpc_plan"
    inputs: List[str] = []
    outputs: List[str] = ["next_action"]

    def __init__(self) -> None:
        self._ensemble = None
        allowed = {"retrieve", "aggregate", "llama_generate"}
        self._action_candidates = [
            action for action in candidate_actions() if str(action.get("name") or "") in allowed
        ]
        self._action_space = list(DEFAULT_ACTION_SPACE)

    def run(self, ctx: Context) -> Result:
        data = ctx.setdefault("data", {})
        model_path_str = data.get("dynamics_model_path") or data.get("dynamics_model") or str(DEFAULT_MODEL_PATH)
        model_path = Path(model_path_str)

        if torch is not None and (self._ensemble is None or getattr(self._ensemble, "_loaded_path", None) != model_path):
            if model_path.exists():
                self._ensemble = load_ensemble(str(model_path))
                setattr(self._ensemble, "_loaded_path", model_path)
            else:
                self._ensemble = None

        state = compact_state(data)
        state_vec = state_to_vector(state)

        if torch is None or not self._ensemble:
            action = self._fallback_action(data)
            data["next_action"] = action
            data["_last_action"] = action
            data["mpc_mode"] = "heuristic"
            return {
                "ok": True,
                "output": {"next_action": action},
                "logs": ["mpc_plan: ensemble unavailable, using fallback action"],
                "checks": {},
                "reward": 0.0,
                "effects": ["next_action_ready"],
            }

        best_score = float("-inf")
        best_variance = float("inf")
        best_sequence: List[Dict[str, object]] = []
        base_horizon = max(2, int(data.get("mpc_rollout_horizon") or 2))
        horizon_options = [2, 3]
        if base_horizon not in horizon_options:
            horizon_options.append(base_horizon)
        candidates = int(data.get("mpc_rollout_candidates") or 12)
        penalty = float(data.get("mpc_uncertainty_penalty") or 0.8)
        for _ in range(candidates):
            horizon = random.choice(horizon_options)
            sequence = [random.choice(self._action_candidates) for _ in range(horizon)]
            score, variance = self._evaluate_sequence(state_vec, sequence, penalty)
            if score > best_score:
                best_score = score
                best_sequence = sequence
                best_variance = variance

        if not best_sequence:
            best_sequence = [self._fallback_action(data)]
        variance_threshold = float(data.get("mpc_variance_threshold") or 1.75)
        if best_variance > variance_threshold:
            action = self._fallback_action(data)
            data["next_action"] = action
            data["_last_action"] = action
            data["mpc_mode"] = "heuristic"
            return {
                "ok": True,
                "output": {"next_action": action},
                "logs": [f"mpc_plan: variance {best_variance:.3f} above threshold, using fallback"],
                "checks": {},
                "reward": 0.0,
                "effects": ["next_action_ready"],
            }
        next_action = best_sequence[0]
        data["next_action"] = next_action
        data["_last_action"] = next_action
        data["mpc_mode"] = "ensemble"

        return {
            "ok": True,
            "output": {"next_action": next_action},
            "logs": [
                f"mpc_plan: selected action {next_action['name']} (score={best_score:.3f}, variance={best_variance:.3f})"
            ],
            "checks": {},
            "reward": best_score,
            "effects": ["next_action_ready"],
        }

    def _evaluate_sequence(
        self,
        state_vec: List[float],
        sequence: Sequence[Dict[str, object]],
        penalty: float,
    ) -> Tuple[float, float]:
        if torch is None or self._ensemble is None:
            return float("-inf"), float("inf")
        assert self._ensemble is not None
        current_state = torch.tensor(state_vec, dtype=torch.float32)
        total_score = 0.0
        max_variance = 0.0
        for action in sequence:
            action_name = str(action.get("name") or "")
            action_vec = action_to_vector(action_name, self._action_space)
            mean_next, variance = self._ensemble.predict(current_state.tolist(), action_vec)
            if hasattr(variance, "mean"):
                var_value = float(variance.mean().item())
            else:
                var_value = float(variance)
            max_variance = max(max_variance, var_value)
            reward_estimate = self._state_reward(mean_next)
            total_score += reward_estimate - penalty * var_value
            current_state = mean_next
        return total_score, max_variance

    def _state_reward(self, state: torch.Tensor) -> float:
        if not isinstance(state, torch.Tensor):
            return 0.0
        values = state.tolist()
        has_sources = float(values[0]) if len(values) > 0 else 0.0
        passages_norm = float(values[1]) if len(values) > 1 else 0.0
        recency_norm = float(values[2]) if len(values) > 2 else 0.0
        tokens_norm = float(values[3]) if len(values) > 3 else 0.0
        return 0.6 * has_sources + 0.2 * passages_norm + 0.1 * recency_norm - 0.2 * tokens_norm

    def _fallback_action(self, data: Mapping[str, Any]) -> Dict[str, object]:
        if not isinstance(data.get("passages"), list) or not data.get("passages"):
            return {
                "name": "retrieve",
                "args": {
                    "k_passages": int(data.get("k_passages") or 12),
                    "fresh_days": data.get("fresh_days") or 0,
                },
            }
        if not data.get("aggregated_text"):
            return {
                "name": "aggregate",
                "args": {
                    "aggregate_strategy": str(data.get("aggregate_strategy") or "concatenate"),
                },
            }
        return {
            "name": "llama_generate",
            "args": {"profile": str(data.get("llama_profile") or "research_plan")},
        }


__all__ = ["MPCPlan"]
