from __future__ import annotations

from typing import List, Optional

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from core.config import load_config
from core.planner import normalise_goal_flags
from core.meta_controller import choose_bundle, record_bundle_outcome
from core import planner as planner_module
from core.rewards import ensure_reward_dict
from main import build_runtime
from tools.registry import TOOLS as TOOL_REGISTRY

app = FastAPI(title="Super AI Server")


class PlanStep(BaseModel):
    behavior: str
    effects: List[str]
    rewards: dict
    rationale: dict


class AskRequest(BaseModel):
    question: str
    config_path: Optional[str] = None
    k_passages: Optional[int] = None
    max_chars: Optional[int] = None
    tags: Optional[str] = None
    index_backend: Optional[str] = None
    cited: Optional[bool] = None
    grounded: Optional[bool] = None
    verbose: Optional[bool] = None
    max_verbose: Optional[bool] = None
    min_words: Optional[int] = None
    max_words: Optional[int] = None
    fresh: Optional[int] = None
    explain: Optional[bool] = None
    no_bandit: Optional[bool] = None


class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    plan: List[PlanStep]
    rewards: dict
    goal_satisfied: bool
    expansions: int


class PlanRequest(BaseModel):
    goal: str
    text: Optional[str] = None
    policies: Optional[str] = None
    config_path: Optional[str] = None
    max_expansions: Optional[int] = None
    explain: Optional[bool] = None
    no_bandit: Optional[bool] = None


class PlanResponse(BaseModel):
    plan: List[PlanStep]
    rewards: dict
    goal_satisfied: bool
    expansions: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _build_plan_steps(steps) -> List[PlanStep]:
    return [
        PlanStep(
            behavior=behavior,
            effects=info.get("effects", []),
            rewards=ensure_reward_dict(info.get("rewards", {})),
            rationale=info.get("rationale", {}),
        )
        for behavior, info in steps
    ]


@app.post("/ask", response_model=AskResponse)
def api_ask(req: AskRequest) -> AskResponse:
    config = load_config(req.config_path)
    registry, interpreter, router = build_runtime()

    ask_cfg = config.get("ask", {}) if isinstance(config.get("ask"), dict) else {}

    index_backend = req.index_backend or ask_cfg.get("index_backend", "hnsw")
    k_passages = req.k_passages if req.k_passages is not None else ask_cfg.get("k_passages", 12)
    max_chars = req.max_chars if req.max_chars is not None else ask_cfg.get("max_chars", 12000)
    tags = req.tags if req.tags is not None else ask_cfg.get("tags", "")
    fresh_days = req.fresh if req.fresh is not None else ask_cfg.get("fresh_days")
    explain = bool(req.explain or ask_cfg.get("explain", False))
    no_bandit = bool(req.no_bandit or ask_cfg.get("no_bandit", False))
    max_expansions = ask_cfg.get("max_expansions", 25)

    if req.max_verbose:
        verbosity = "max"
    elif req.verbose:
        verbosity = "verbose"
    else:
        verbosity = str(ask_cfg.get("verbosity", "verbose"))

    if req.min_words is not None:
        min_words = req.min_words
    else:
        cfg_min = ask_cfg.get("min_words")
        if cfg_min is not None:
            min_words = cfg_min
        else:
            min_words = 1500 if verbosity == "max" else 800 if verbosity == "verbose" else 400

    max_words = req.max_words if req.max_words is not None else ask_cfg.get("max_words")

    goal_terms: List[str] = []
    if req.cited or ask_cfg.get("cited", False):
        goal_terms.append("cited")
    if req.grounded or ask_cfg.get("grounded", False):
        goal_terms.append("grounded")
    if (
        req.verbose
        or req.max_verbose
        or req.min_words is not None
        or verbosity in {"verbose", "max"}
        or ask_cfg.get("min_words")
    ):
        goal_terms.append("verbose")
    if fresh_days:
        goal_terms.append("fresh")
    if not goal_terms:
        goal_terms = ["cited", "grounded", "verbose"]

    goal_str = ",".join(goal_terms)
    ctx = {
        "text": req.question,
        "data": {
            "text": req.question,
            "question": req.question,
            "index_backend": index_backend,
            "k_passages": k_passages,
            "max_chars": max_chars,
            "tags": tags,
            "verbosity": verbosity,
            "min_words": min_words,
        },
        "router": {"goal_text": goal_str, "no_bandit": no_bandit},
    }
    ctx["data"].setdefault("tools_registry", TOOL_REGISTRY)
    if max_words:
        ctx["data"]["max_words"] = max_words
    if fresh_days:
        ctx["data"]["fresh_days"] = fresh_days
    if "grounded" in goal_terms or "cited" in goal_terms:
        ctx["data"]["apply_research_prompt"] = True
    else:
        ctx["data"].setdefault("allow_ungrounded_profile", True)
    cluster = router.cluster_hint(goal_str, ctx)
    ctx["router"]["cluster_bias"] = cluster

    goal_flags = normalise_goal_flags(goal_str)
    choose_bundle(
        ctx,
        planners=["planner_react", "planner_tot", "planner_fp"],
        judge_bundle=ctx["data"].get("judge_bundle"),
        budget=max_expansions,
    )
    plan_result = planner_module.plan(
        goal_flags,
        ctx,
        registry,
        interpreter,
        cluster_bias=cluster,
        max_expansions=max_expansions,
    )

    steps = plan_result.get("steps", [])
    action_steps = [step for step in steps if step[0] != "meaning_infer"]
    final_ctx = plan_result.get("ctx", ctx)
    final_data = final_ctx.get("data", {})

    if not action_steps:
        fallback_ctx = copy.deepcopy(ctx)
        summary_result = interpreter.execute("summarize", fallback_ctx)
        if not summary_result.get("ok"):
            remaining = plan_result.get("remaining_flags") or goal_flags
            raise HTTPException(
                status_code=404,
                detail=f"No plan found. Remaining goals: {', '.join(remaining)}",
            )

        summary = fallback_ctx.get("data", {}).get("summary", "")
        rewards = ensure_reward_dict(summary_result.get("rewards", {}))
        plan_entry = PlanStep(
            behavior="summarize",
            effects=summary_result.get("effects", []),
            rewards=rewards,
            rationale=summary_result.get("rationale", {}),
        )
        return AskResponse(
            answer=summary,
            sources=[],
            plan=[plan_entry],
            rewards=rewards,
            goal_satisfied=True,
            expansions=int(plan_result.get("expansions", 0)),
        )

    final_rewards = ensure_reward_dict(steps[-1][1].get("rewards", {}))
    answer = final_data.get("answer") or final_data.get("summary") or final_data.get("aggregated_text") or ""
    if not answer:
        remaining = plan_result.get("remaining_flags") or goal_flags
        raise HTTPException(
            status_code=404,
            detail=f"No plan output produced. Remaining goals: {', '.join(remaining)}",
        )
    raw_sources = final_data.get("sources", []) or []
    normalized_sources: List[str] = []
    for entry in raw_sources:
        if isinstance(entry, str):
            normalized_sources.append(entry)
            continue
        if isinstance(entry, dict):
            marker = str(entry.get("marker") or "").strip()
            title = str(entry.get("source") or entry.get("title") or "").strip()
            url = str(entry.get("url") or "").strip()
            if marker and url:
                label = title or url
                normalized_sources.append(f"{marker} {label} - {url}")
            elif title:
                normalized_sources.append(title)
            else:
                normalized_sources.append(marker or url)
            continue
        normalized_sources.append(str(entry))
    sources = [source for source in normalized_sources if source]

    record_bundle_outcome(final_ctx, final_rewards)

    return AskResponse(
        answer=answer,
        sources=sources,
        plan=_build_plan_steps(steps),
        rewards=final_rewards,
        goal_satisfied=bool(plan_result.get("goal_satisfied", False)),
        expansions=int(plan_result.get("expansions", 0)),
    )


@app.post("/plan", response_model=PlanResponse)
def api_plan(req: PlanRequest) -> PlanResponse:
    config = load_config(req.config_path)
    registry, interpreter, router = build_runtime()
    plan_cfg = config.get("plan", {}) if isinstance(config.get("plan"), dict) else {}

    ctx = {
        "text": req.text or "",
        "data": {
            "text": req.text or "",
        },
        "router": {
            "goal_text": req.goal,
            "no_bandit": bool(req.no_bandit or plan_cfg.get("no_bandit", False)),
        },
    }
    ctx["data"].setdefault("tools_registry", TOOL_REGISTRY)
    if req.text:
        ctx["data"]["text"] = req.text
    if req.policies:
        ctx["data"]["policies"] = [item.strip() for item in req.policies.split(',') if item.strip()]

    cluster = router.cluster_hint(req.goal, ctx)
    ctx["router"]["cluster_bias"] = cluster

    goal_flags = normalise_goal_flags(req.goal)
    choose_bundle(
        ctx,
        planners=["planner_react", "planner_tot", "planner_fp"],
        judge_bundle=ctx["data"].get("judge_bundle"),
        budget=req.max_expansions or plan_cfg.get("max_expansions", 20),
    )
    plan_result = planner_module.plan(
        goal_flags,
        ctx,
        registry,
        interpreter,
        cluster_bias=cluster,
        max_expansions=req.max_expansions or plan_cfg.get("max_expansions", 20),
    )

    steps = plan_result.get("steps", [])
    action_steps = [step for step in steps if step[0] != "meaning_infer"]
    if not action_steps:
        remaining = plan_result.get("remaining_flags") or goal_flags
        raise HTTPException(
            status_code=404,
            detail=f"No plan found. Remaining goals: {', '.join(remaining)}",
        )

    final_ctx = plan_result.get("ctx", ctx)
    final_rewards = ensure_reward_dict(steps[-1][1].get("rewards", {}))
    record_bundle_outcome(final_ctx, final_rewards)
    if not req.text:
        final_data = final_ctx.get("data", {})
        summary_text = final_data.get("summary")
        meaningful_steps = [step for step in action_steps if step[0] != "summarize"]
        if not meaningful_steps and not (summary_text and summary_text.strip()):
            remaining = plan_result.get("remaining_flags") or goal_flags
            raise HTTPException(
                status_code=404,
                detail=f"No plan found. Remaining goals: {', '.join(remaining)}",
            )

    return PlanResponse(
        plan=_build_plan_steps(steps),
        rewards=final_rewards,
        goal_satisfied=bool(plan_result.get("goal_satisfied", False)),
        expansions=int(plan_result.get("expansions", 0)),
    )

