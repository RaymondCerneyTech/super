from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core.bandit import UCB1
from core.cues import select_cue
from core.interfaces import Context
from core.rewards import ensure_reward_dict
from core import llama_profiles

STATE_ENV = "SUPER_BUNDLE_STATE"
DEFAULT_STATE_PATH = Path("data") / "bandit" / "bundle_state.json"
BUNDLE_LOG_PATH = Path("logs") / "bundles.jsonl"
BUNDLE_BANDITS: Dict[str, UCB1] = {}
STATE_PATH = Path(os.getenv(STATE_ENV, "")).expanduser() if os.getenv(STATE_ENV) else DEFAULT_STATE_PATH
KNOWN_PLANNERS: Tuple[str, ...] = ("planner_react", "planner_tot", "planner_fp", "planner_analogy", "planner_causal")


def choose_bundle(
    ctx: Context,
    *,
    planners: Sequence[str] | None = None,
    judge_bundle: Optional[Iterable[str]] = None,
    budget: Optional[int] = None,
) -> Dict[str, Any]:
    planner_list = list(planners or [])
    cue = _infer_cue(ctx)
    data_layer = ctx.get("data") if isinstance(ctx, dict) else {}
    if not isinstance(data_layer, dict):
        data_layer = {}
    max_models = _coerce_positive(data_layer.get("max_models"))
    max_planners_limit = _coerce_positive(data_layer.get("max_planners"))
    max_candidates_limit = _coerce_positive(data_layer.get("max_candidates"))
    candidate_cap = budget or max_candidates_limit
    planner_candidates = _planner_candidates(planner_list, cue, candidate_cap, max_planners_limit)
    judge_candidates = _judge_candidates(judge_bundle)
    model_candidates = _model_candidates(cue, max_models)
    bundles = _cartesian_bundles(model_candidates, planner_candidates, judge_candidates, cue)
    if not bundles:
        fallback = {
            "models": [],
            "planners": planner_list or ["planner_fp"],
            "judges": judge_candidates[0] if judge_candidates else ["meta_judge"],
            "arm_id": _bundle_id([], planner_list or ["planner_fp"], judge_candidates[0] if judge_candidates else ["meta_judge"], cue),
            "cue": cue,
        }
        bundles = [fallback]

    bandit = BUNDLE_BANDITS.setdefault(cue, UCB1())
    for entry in bundles:
        bandit.ensure_arm(entry["arm_id"])
    selected_arm = bandit.pick() or bundles[0]["arm_id"]
    selected = next((entry for entry in bundles if entry["arm_id"] == selected_arm), bundles[0])

    router_state = ctx.setdefault("router", {}) if isinstance(ctx, dict) else {}
    if isinstance(router_state, dict):
        router_state["bundle_arm"] = selected["arm_id"]
        router_state["bundle_cue"] = cue
        router_state["bundle_candidates"] = [entry["arm_id"] for entry in bundles]
        router_state["bundle_components"] = {
            "models": selected["models"],
            "planners": selected["planners"],
            "judges": selected["judges"],
        }

    data = ctx.setdefault("data", {}) if isinstance(ctx, dict) else {}
    if isinstance(data, dict):
        data["model_bundle"] = list(selected["models"])
        data["planner_bundle"] = list(selected["planners"])
        data["judge_bundle"] = list(selected["judges"])
        data["bundle_arm"] = selected["arm_id"]
        if selected["models"] and not data.get("llama_profile"):
            data["llama_profile"] = selected["models"][0]

    return selected


def record_bundle_outcome(ctx: Context, rewards: Dict[str, float] | float) -> None:
    if not isinstance(ctx, dict):
        return
    router_state = ctx.get("router")
    data = ctx.get("data") if isinstance(ctx.get("data"), dict) else {}
    arm = None
    cue = None
    if isinstance(router_state, dict):
        arm = router_state.get("bundle_arm")
        cue = router_state.get("bundle_cue")
    if not cue:
        cue = _infer_cue(ctx)
    if not arm and isinstance(data, dict):
        arm = data.get("bundle_arm")
    if not arm or not cue:
        return
    bandit = BUNDLE_BANDITS.setdefault(cue, UCB1())
    bandit.ensure_arm(arm)
    reward_dict = ensure_reward_dict(rewards)
    reward_value = reward_dict.get("verifier", reward_dict.get("overall", 0.0))
    bandit.record(arm, reward_value)
    _persist_state()
    if isinstance(router_state, dict):
        router_state["bundle_stats"] = {
            "arm": arm,
            "cue": cue,
            "reward": reward_value,
        }
    log_entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "arm": arm,
        "cue": cue,
        "reward": reward_value,
        "planners": (data or {}).get("planner_bundle"),
        "models": (data or {}).get("model_bundle"),
        "judges": (data or {}).get("judge_bundle"),
        "goal": (data or {}).get("goal"),
    }
    _append_bundle_log(log_entry)


def _planner_candidates(
    planners: Sequence[str],
    cue: str,
    max_candidates: Optional[int],
    limit: Optional[int],
) -> List[List[str]]:
    provided = _dedupe(planners)
    unique = provided or [planner for planner in KNOWN_PLANNERS]
    if not unique:
        unique = ["planner_fp"]
    effective_cap = max_candidates or len(unique)
    if limit:
        effective_cap = min(effective_cap, limit)
    capacity = max(1, min(len(unique), effective_cap, 3))
    primary = unique[:capacity]
    candidates: List[List[str]] = [primary]

    reactive = [p for p in ("planner_react", "planner_tot") if p in unique]
    if reactive and reactive not in candidates:
        candidates.append(reactive)

    fallback = [p for p in ("planner_fp", "planner_causal") if p in unique]
    if fallback and fallback not in candidates:
        candidates.append(fallback)

    if len(unique) > 1:
        candidates.append([unique[0]])

    deduped: List[List[str]] = []
    for bundle in candidates:
        if bundle and bundle not in deduped:
            deduped.append(bundle)
    return deduped or [primary]


def _judge_candidates(judges: Optional[Iterable[str]]) -> List[List[str]]:
    candidates: List[List[str]] = []
    provided = [str(j).strip() for j in (judges or []) if str(j).strip()]
    if provided:
        candidates.append(provided)
    candidates.append(["meta_judge"])
    candidates.append(["meta_judge", "consistency"])
    deduped: List[List[str]] = []
    for bundle in candidates:
        key = tuple(bundle)
        if key not in [tuple(existing) for existing in deduped]:
            deduped.append(bundle)
    return deduped


def _model_candidates(cue: str, limit: Optional[int] = None) -> List[List[str]]:
    profiles = [name for name, _ in llama_profiles.list_profiles()]
    options: List[List[str]] = []

    def add_if_available(name: str) -> None:
        if name in profiles and [name] not in options:
            options.append([name])

    if cue in {"summarize", "policy", "plan"}:
        add_if_available("research_plan")
    if cue == "numbers":
        add_if_available("math_solver")
    if cue == "compose":
        add_if_available("poem")
    if profiles:
        add_if_available(profiles[0])
    if not options:
        options.append([])
    if limit and limit > 0:
        options = options[:limit]
    return options


def _cartesian_bundles(
    models: List[List[str]],
    planners: List[List[str]],
    judges: List[List[str]],
    cue: str,
) -> List[Dict[str, Any]]:
    bundles: List[Dict[str, Any]] = []
    for model_bundle, planner_bundle, judge_bundle in product(models, planners, judges):
        arm_id = _bundle_id(model_bundle, planner_bundle, judge_bundle, cue)
        bundles.append(
            {
                "models": list(model_bundle),
                "planners": list(planner_bundle),
                "judges": list(judge_bundle),
                "arm_id": arm_id,
                "cue": cue,
            }
        )
    return bundles


def _bundle_id(models: Sequence[str], planners: Sequence[str], judges: Sequence[str], cue: str) -> str:
    model_str = "+".join(models) or "none"
    planner_str = "+".join(planners) or "none"
    judge_str = "+".join(judges) or "none"
    return f"{model_str}|{planner_str}|{judge_str}|{cue or 'generic'}"


def _dedupe(items: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _infer_cue(ctx: Context) -> str:
    router_state = ctx.get("router") if isinstance(ctx, dict) else {}
    goal_text = ""
    if isinstance(router_state, dict):
        goal_text = str(router_state.get("goal_text") or "")
    data = ctx.get("data") if isinstance(ctx, dict) else {}
    if isinstance(data, dict):
        goal_text = str(data.get("goal") or goal_text)
    text = ""
    if isinstance(ctx, dict):
        text = str(ctx.get("text") or "")
    if isinstance(data, dict):
        text = str(data.get("text") or text)
    combined = " ".join(part for part in [goal_text, text] if part)
    cue = select_cue(combined)
    return cue or "compose"


def _coerce_positive(value: Any) -> Optional[int]:
    try:
        val = int(value)
    except (TypeError, ValueError):
        return None
    return val if val > 0 else None


def _resolve_state_payload() -> Dict[str, Any]:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def _persist_state() -> None:
    if not STATE_PATH:
        return
    payload = {cue: bandit.to_dict() for cue, bandit in BUNDLE_BANDITS.items()}
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        return


def _load_state() -> None:
    payload = _resolve_state_payload()
    if not payload:
        return
    for cue, state in payload.items():
        if not isinstance(state, dict):
            continue
        bandit = UCB1()
        bandit.load_state(state)
        BUNDLE_BANDITS[cue] = bandit


_load_state()


def _append_bundle_log(entry: Dict[str, Any]) -> None:
    try:
        BUNDLE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with BUNDLE_LOG_PATH.open("a", encoding="utf-8") as handle:
            json.dump(entry, handle)
            handle.write("\n")
    except OSError:
        return


__all__ = ["choose_bundle", "record_bundle_outcome"]
