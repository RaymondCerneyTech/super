from __future__ import annotations

import difflib
import json
import math
import re
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Tuple, Union

try:  # optional dependency
    import jsonschema  # type: ignore
except ImportError:  # pragma: no cover - optional
    jsonschema = None  # type: ignore

try:  # optional dependency
    import sympy  # type: ignore
except ImportError:  # pragma: no cover - optional
    sympy = None  # type: ignore

Number = Union[int, float]
RewardLike = Union[Number, Mapping[str, Any], None]

EXPLANATION_WEIGHTS = {
    "explanation_presence": 0.02,
    "explanation_specificity": 0.02,
    "explanation_alignment": 0.01,
}
SENTENCE_RE = re.compile(r"[^\n]+")
TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def ensure_reward_dict(value: RewardLike) -> Dict[str, float]:
    cleaned: Dict[str, float] = {}
    if isinstance(value, Mapping):
        for key, raw in value.items():
            val = _coerce_float(raw)
            if val is None:
                continue
            cleaned[str(key)] = val
    elif isinstance(value, (int, float)):
        val = _coerce_float(value)
        if val is not None:
            cleaned["overall"] = val
    if not cleaned:
        cleaned["overall"] = 0.0
    if "overall" not in cleaned:
        extras = [val for key, val in cleaned.items() if key != "overall"]
        cleaned["overall"] = sum(extras) / len(extras) if extras else 0.0
    return cleaned


def aggregate_reward(reward: RewardLike) -> float:
    reward_dict = ensure_reward_dict(reward)
    extras = [value for key, value in reward_dict.items() if key != "overall"]
    if extras:
        return sum(extras) / len(extras)
    return reward_dict.get("overall", 0.0)


def merge_rewards(target: MutableMapping[str, float], updates: Iterable[Tuple[str, Any]]) -> None:
    if target is None:
        return
    for key, raw in updates:
        if key is None:
            continue
        val = _coerce_float(raw)
        if val is None:
            continue
        name = str(key)
        target[name] = target.get(name, 0.0) + val
    target.setdefault("overall", 0.0)


def apply_explanation_bonus(rewards: MutableMapping[str, float], scores: Mapping[str, Any]) -> MutableMapping[str, float]:
    if not scores:
        return rewards
    rewards = rewards or {"overall": 0.0}
    bonus = 0.0
    for key, weight in EXPLANATION_WEIGHTS.items():
        val = _coerce_float(scores.get(key))
        clamped = _clamp(val if val is not None else 0.0)
        rewards[key] = clamped
        bonus += clamped * weight
    rewards["overall"] = _clamp(rewards.get("overall", 0.0) + bonus)
    return rewards


def compute_explanation_scores(rationale: Any, output: Mapping[str, Any]) -> Dict[str, float]:
    if not isinstance(rationale, Mapping):
        return {"explanation_presence": 0.0, "explanation_specificity": 0.0, "explanation_alignment": 0.0}
    why = str(rationale.get("why", "") or "").strip()
    evidence_obj = rationale.get("evidence")
    if isinstance(evidence_obj, str):
        evidence_list = [evidence_obj.strip()]
    elif isinstance(evidence_obj, Iterable):
        evidence_list = [str(item).strip() for item in evidence_obj if str(item).strip()]
    else:
        evidence_list = []
    presence = 1.0 if (why or evidence_list) else 0.0
    specificity = _clamp(len(evidence_list) / 3) if evidence_list else presence * 0.5
    output_text = _to_text(output)
    output_lower = output_text.lower()
    if evidence_list and output_lower:
        matched = sum(1 for entry in evidence_list if entry.lower() in output_lower)
        alignment = _clamp(matched / len(evidence_list))
    else:
        alignment = presence if output_lower else 0.0
    return {
        "explanation_presence": presence,
        "explanation_specificity": specificity,
        "explanation_alignment": alignment,
    }


def citation_coverage(text: Any) -> float:
    content = _to_text(text)
    lines = [line.strip() for line in re.split(r"[\r\n]+", content) if line.strip()]
    if not lines:
        return 0.0
    cited = sum(1 for line in lines if "[S" in line or "[s" in line)
    return _clamp(cited / len(lines))


def faithfulness(answer: Any, context: Any) -> float:
    answer_text = _to_text(answer)
    context_text = _to_text(context)
    if not answer_text.strip() or not context_text.strip():
        return 0.0
    answer_tokens = set(_tokenize(answer_text))
    context_tokens = set(_tokenize(context_text))
    if not answer_tokens or not context_tokens:
        return 0.0
    overlap = len(answer_tokens & context_tokens)
    ratio = overlap / max(1, len(answer_tokens))
    return _clamp(ratio)


def verbosity_score(text: Any, target_words: int) -> float:
    words = len(_tokenize(_to_text(text)))
    target = max(1, int(target_words or 0))
    ratio = words / target
    return _clamp(ratio)


def similarity_step(target: Any, action: Any) -> float:
    target_text = str(target or "").strip()
    action_text = str(action or "").strip()
    if not target_text and not action_text:
        return 1.0
    matcher = difflib.SequenceMatcher(None, target_text, action_text)
    return _clamp(matcher.ratio())


def json_schema_ok(text: Any, schema: Any) -> float:
    if not schema:
        return 0.0
    try:
        payload = json.loads(text if isinstance(text, str) else json.dumps(text))
    except (TypeError, json.JSONDecodeError):
        return 0.0
    if jsonschema is None:
        required = schema.get("required", [])
        if not required:
            return 1.0
        payload_keys = set(payload.keys()) if isinstance(payload, dict) else set()
        return 1.0 if all(key in payload_keys for key in required) else 0.0
    try:  # pragma: no cover - depends on optional lib
        jsonschema.validate(payload, schema)
        return 1.0
    except Exception:
        return 0.0


def math_exact(pred: Any, target: Any, *, tol: float = 1e-6) -> float:
    if sympy is not None:  # pragma: no cover - optional branch
        try:
            pred_expr = sympy.simplify(pred)
            target_expr = sympy.simplify(target)
            return 1.0 if sympy.simplify(pred_expr - target_expr) == 0 else 0.0
        except Exception:
            pass
    try:
        pred_val = float(pred)
        target_val = float(target)
        return 1.0 if abs(pred_val - target_val) <= tol else 0.0
    except (TypeError, ValueError):
        return 1.0 if str(pred).strip() == str(target).strip() else 0.0


def unit_tests_pass(report: Any) -> float:
    if report is None:
        return 0.0
    if isinstance(report, str):
        try:
            report = json.loads(report)
        except json.JSONDecodeError:
            report = report.strip().splitlines()
    if isinstance(report, dict):
        passed = float(report.get("passed") or report.get("success") or 0.0)
        total = float(report.get("total") or (passed + report.get("failed", 0)))
        if total <= 0:
            return 0.0
        return _clamp(passed / total)
    if isinstance(report, (list, tuple)):
        total = len(report)
        if total == 0:
            return 0.0
        passed = sum(1 for item in report if bool(item))
        return _clamp(passed / total)
    return 0.0


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(_to_text(item) for item in value)
    if isinstance(value, dict):
        if "text" in value:
            return _to_text(value["text"])
        return "\n".join(_to_text(item) for item in value.values())
    return str(value)


def _tokenize(text: str) -> Tuple[str, ...]:
    return tuple(match.group(0).lower() for match in TOKEN_RE.finditer(text))


__all__ = [
    "ensure_reward_dict",
    "aggregate_reward",
    "merge_rewards",
    "apply_explanation_bonus",
    "compute_explanation_scores",
    "citation_coverage",
    "faithfulness",
    "verbosity_score",
    "similarity_step",
    "json_schema_ok",
    "math_exact",
    "unit_tests_pass",
]
