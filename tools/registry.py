from __future__ import annotations

import re
from typing import Any, Dict, List

from tools.aci import (
    append_file as aci_append_file,
    grep as aci_grep,
    grep_repo as aci_grep_repo,
    read_file as aci_read_file,
    run_pytest as aci_run_pytest,
    write_file as aci_write_file,
)
from tools.data_utils import csv_summary, dedupe_lines, json_query, table_detect
from tools.math_units import math_eval_safe, unit_convert_basic
from tools.memory_tools import cache_get, cache_put, reflection_add, reflection_get
from tools.planning import causal_dot_builder, timeline_normalize
from tools.web import extract_facts, extract_links, html_to_text, web_get


def _clean(text: str) -> str:
    return " ".join(str(text or "").split())


def rewrite_clarity(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = _clean(payload.get("text") or payload.get("draft") or "")
    return {"text": text, "quality_gain": 0.10, "faithfulness": 1.0, "notes": "clarity rewrite"}


def rewrite_style(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = _clean(payload.get("text") or payload.get("draft") or "")
    tone = str(payload.get("tone") or "neutral").lower()
    if tone == "friendly":
        text = text.replace(" do not ", " don't ").replace(" cannot ", " can't ")
    elif tone == "formal":
        text = re.sub(
            r"\b(can't|don't|won't)\b",
            lambda m: {"can't": "cannot", "don't": "do not", "won't": "will not"}[m.group(0)],
            text,
        )
    return {"text": text, "quality_gain": 0.08, "faithfulness": 1.0, "notes": f"style={tone}"}


def summarize_bullets(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = _clean(payload.get("text") or payload.get("draft") or "")
    sents = re.split(r"(?<=[.!?])\s+", text)
    bullets = sents[:5]
    out = "- " + "\n- ".join([s for s in bullets if s])
    return {"text": out, "quality_gain": 0.12, "faithfulness": 0.9, "notes": "5-bullet summary"}


_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def numbers_guard(payload: Dict[str, Any]) -> Dict[str, Any]:
    src = _clean(payload.get("source_text") or payload.get("text") or payload.get("draft") or "")
    dst = _clean(payload.get("text") or "")
    src_nums = _NUM_RE.findall(src)
    dst_nums = _NUM_RE.findall(dst)
    faithful = 1.0 if src_nums == dst_nums else 0.0
    notes = "numbers ok" if faithful == 1.0 else f"numbers changed: {src_nums} -> {dst_nums}"
    return {"text": dst or src, "quality_gain": 0.0, "faithfulness": faithful, "notes": notes}


_PI_PAT = re.compile(r"\b(pass(word)?|api[_-]?key|secret|ssn|credit\s*card)\b", re.I)


def policy_check_lite(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = _clean(payload.get("text") or payload.get("draft") or "")
    hits = _PI_PAT.findall(text)
    notes = "policy ok" if not hits else f"policy risk: {len(hits)} hit(s)"
    return {"text": text, "quality_gain": 0.0, "faithfulness": 1.0, "notes": notes}


_POS = {"good", "great", "excellent", "love", "win", "happy", "benefit", "improve", "success"}
_NEG = {"bad", "terrible", "awful", "hate", "fail", "sad", "risk", "worse", "problem"}


def sentiment_label(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = _clean(payload.get("text") or payload.get("draft") or "")
    tokens = set(re.findall(r"[A-Za-z']+", text.lower()))
    pos = len(tokens & _POS)
    neg = len(tokens & _NEG)
    label = "neutral"
    if pos > neg:
        label = "positive"
    elif neg > pos:
        label = "negative"
    return {"text": text, "quality_gain": 0.0, "faithfulness": 1.0, "notes": f"sentiment={label}"}


def extract_actions(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = _clean(payload.get("text") or payload.get("draft") or "")
    actions: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if re.match(r"^(add|create|update|remove|check|verify|run|write|summarize|plan)\b", line, re.I):
            actions.append(line)
    if not actions:
        sents = re.split(r"(?<=[.!?])\s+", text)
        actions = [
            s
            for s in sents
            if re.match(r"^(Add|Create|Update|Remove|Check|Verify|Run|Write|Summarize|Plan)\b", s)
        ]
    out = "- " + "\n- ".join(actions) if actions else "- (no clear actions found)"
    return {"text": out, "quality_gain": 0.06, "faithfulness": 1.0, "notes": "extracted actions"}


TOOLS: Dict[str, Dict[str, Any]] = {
    "web_get": {"name": "web_get", "affordances": ["web", "fetch"], "fn": web_get},
    "html_to_text": {"name": "html_to_text", "affordances": ["web", "clean"], "fn": html_to_text},
    "extract_links": {"name": "extract_links", "affordances": ["web", "analyze"], "fn": extract_links},
    "extract_facts": {"name": "extract_facts", "affordances": ["web", "analyze"], "fn": extract_facts},
    "table_detect": {"name": "table_detect", "affordances": ["data", "parse"], "fn": table_detect},
    "csv_summary": {"name": "csv_summary", "affordances": ["data", "summarize"], "fn": csv_summary},
    "json_query": {"name": "json_query", "affordances": ["data", "query"], "fn": json_query},
    "dedupe_lines": {"name": "dedupe_lines", "affordances": ["data", "clean"], "fn": dedupe_lines},
    "math_eval_safe": {"name": "math_eval_safe", "affordances": ["math", "compute"], "fn": math_eval_safe},
    "unit_convert_basic": {
        "name": "unit_convert_basic",
        "affordances": ["math", "units"],
        "fn": unit_convert_basic,
    },
    "causal_dot_builder": {
        "name": "causal_dot_builder",
        "affordances": ["plan", "causal"],
        "fn": causal_dot_builder,
    },
    "timeline_normalize": {
        "name": "timeline_normalize",
        "affordances": ["plan", "time"],
        "fn": timeline_normalize,
    },
    "read_file": {"name": "read_file", "affordances": ["code", "io"], "fn": aci_read_file},
    "write_file": {"name": "write_file", "affordances": ["code", "io"], "fn": aci_write_file},
    "append_file": {"name": "append_file", "affordances": ["code", "io"], "fn": aci_append_file},
    "grep_repo": {"name": "grep_repo", "affordances": ["code", "search_repo"], "fn": aci_grep_repo},
    "run_pytest": {"name": "run_pytest", "affordances": ["code", "run_tests"], "fn": aci_run_pytest},
    "reflection_add": {
        "name": "reflection_add",
        "affordances": ["learn", "reflect"],
        "fn": reflection_add,
    },
    "reflection_get": {
        "name": "reflection_get",
        "affordances": ["learn", "reflect"],
        "fn": reflection_get,
    },
    "cache_put": {"name": "cache_put", "affordances": ["memo", "cache"], "fn": cache_put},
    "cache_get": {"name": "cache_get", "affordances": ["memo", "cache"], "fn": cache_get},
    "rewrite_clarity": {"name": "rewrite_clarity", "affordances": ["compose", "generic"], "fn": rewrite_clarity},
    "rewrite_style": {"name": "rewrite_style", "affordances": ["compose", "style"], "fn": rewrite_style},
    "summarize_bullets": {
        "name": "summarize_bullets",
        "affordances": ["summarize", "compose"],
        "fn": summarize_bullets,
    },
    "numbers_guard": {"name": "numbers_guard", "affordances": ["numbers", "guard"], "fn": numbers_guard},
    "policy_check_lite": {
        "name": "policy_check_lite",
        "affordances": ["policy", "analyze"],
        "fn": policy_check_lite,
    },
    "sentiment_label": {
        "name": "sentiment_label",
        "affordances": ["sentiment", "analyze"],
        "fn": sentiment_label,
    },
    "extract_actions": {"name": "extract_actions", "affordances": ["plan", "analyze"], "fn": extract_actions},
    # Backwards-compatibility aliases for existing behaviors
    "code_read": {
        "name": "code_read",
        "affordances": ["code_read", "compose"],
        "fn": aci_read_file,
    },
    "code_write": {
        "name": "code_write",
        "affordances": ["code_write", "compose"],
        "fn": aci_write_file,
    },
    "search_repo": {
        "name": "search_repo",
        "affordances": ["search_repo", "plan"],
        "fn": aci_grep,
    },
    "run_tests": {
        "name": "run_tests",
        "affordances": ["run_tests", "plan"],
        "fn": aci_run_pytest,
    },
}


__all__ = [
    "TOOLS",
    "web_get",
    "html_to_text",
    "extract_links",
    "extract_facts",
    "table_detect",
    "csv_summary",
    "json_query",
    "dedupe_lines",
    "math_eval_safe",
    "unit_convert_basic",
    "causal_dot_builder",
    "timeline_normalize",
    "read_file",
    "write_file",
    "append_file",
    "grep_repo",
    "run_pytest",
    "reflection_add",
    "reflection_get",
    "cache_put",
    "cache_get",
    "rewrite_clarity",
    "rewrite_style",
    "summarize_bullets",
    "numbers_guard",
    "policy_check_lite",
    "sentiment_label",
    "extract_actions",
    "aci_read_file",
    "aci_write_file",
    "aci_grep",
    "aci_grep_repo",
    "aci_run_pytest",
]

