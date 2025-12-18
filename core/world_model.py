from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence


DEFAULT_ACTION_SPACE: Sequence[str] = ("retrieve", "aggregate", "llama_generate", "document_formatting")

CANDIDATE_ACTIONS: Sequence[Dict[str, Any]] = (
    {"name": "retrieve", "args": {"k_passages": 5, "fresh_days": 7}},
    {"name": "retrieve", "args": {"k_passages": 12, "fresh_days": 30}},
    {"name": "aggregate", "args": {"aggregate_strategy": "topic_blocks"}},
    {"name": "llama_generate", "args": {"profile": "research_plan"}},
    {"name": "document_formatting", "args": {"format_style": "business"}},
)


def compact_state(data: Mapping[str, Any]) -> Dict[str, Any]:
    meaning = data.get("meaning")
    has_sources = bool(data.get("passages"))
    k_passages = _safe_int(data.get("k_passages"))
    fresh_days = _safe_int(data.get("fresh_days"))
    tokens_out = _count_tokens(data.get("llama_output"))
    profile = None
    if isinstance(data.get("llama"), dict):
        profile = data["llama"].get("profile")
    if not profile:
        profile = data.get("llama_profile")

    state = {
        "meaning": str(meaning) if meaning else "",
        "has_sources": bool(has_sources),
        "k_passages": k_passages,
        "fresh_days": fresh_days,
        "tokens_out": tokens_out,
        "profile": str(profile) if profile else "",
    }
    return state


def state_to_vector(state: Mapping[str, Any]) -> List[float]:
    has_sources = 1.0 if state.get("has_sources") else 0.0
    k_passages = float(state.get("k_passages") or 0) / 12.0
    fresh_days = float(state.get("fresh_days") or 0) / 30.0
    tokens_out = float(state.get("tokens_out") or 0) / 1024.0
    return [has_sources, k_passages, fresh_days, tokens_out]


def action_to_vector(action_name: str, action_space: Sequence[str] | None = None) -> List[float]:
    if action_space is None:
        action_space = DEFAULT_ACTION_SPACE
    vec = [0.0] * len(action_space)
    try:
        index = list(action_space).index(action_name)
        vec[index] = 1.0
    except ValueError:
        pass
    return vec


def extract_action_args(meta: Mapping[str, Any], data: Mapping[str, Any]) -> Dict[str, Any]:
    args_spec = meta.get("args") if isinstance(meta, Mapping) else {}
    snapshot: Dict[str, Any] = {}
    if isinstance(args_spec, Mapping):
        for name in args_spec.keys():
            value = data.get(name)
            if isinstance(value, (str, int, float, bool)) or value is None:
                snapshot[name] = value
            else:
                snapshot[name] = str(value)
    return snapshot


def candidate_actions() -> Sequence[Dict[str, Any]]:
    return CANDIDATE_ACTIONS


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _count_tokens(llama_output: Any) -> int:
    text = ""
    if isinstance(llama_output, Mapping):
        text = str(llama_output.get("text") or "")
    elif isinstance(llama_output, str):
        text = llama_output
    tokens = text.split()
    return len(tokens)


__all__ = [
    "compact_state",
    "state_to_vector",
    "action_to_vector",
    "extract_action_args",
    "candidate_actions",
    "DEFAULT_ACTION_SPACE",
]
