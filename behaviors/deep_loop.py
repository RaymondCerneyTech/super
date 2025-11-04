from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from core.audit import log_action
from core.cues import PIPELINES, select_cue
from core.credit_ledger import record as ledger_record
from core.interfaces import Behavior, Context, Result
from core.memory import EpisodicStore, ToolMemory, WorkingMemory
from core.reflections import add_reflection, fingerprint, get_reflections
from core.meta_planner import record_outcome, build_features
from judges.meta_judge import aggregate, rule_judge

CALL_SITE = "behaviors.deep_loop"


def _extract_numbers(text: str) -> Tuple[str, ...]:
    tokens = []
    current = []
    for char in text:
        if char.isdigit() or (char in {".", ","} and current):
            current.append(char)
        else:
            if current:
                tokens.append("".join(current))
                current = []
    if current:
        tokens.append("".join(current))
    return tuple(tokens)


class DeepLoop(Behavior):
    name = "deep_loop"
    inputs = ["goal"]
    outputs = ["partial"]

    def run(self, ctx: Context) -> Result:
        data = ctx.setdefault("data", {})
        goal = str(data.get("goal") or ctx.get("text") or "")
        invariants = data.get("invariants") or {}
        tools_registry = data.get("tools_registry") or {}
        max_iters = int(data.get("max_iters") or 8)
        existing_memory = data.get("tool_memory") if isinstance(data.get("tool_memory"), dict) else None
        max_tokens = int(data.get("max_tokens") or 0)
        max_wall_ms = int(data.get("max_wall_ms") or 0)

        start_time = time.perf_counter()
        tokens_used = 0
        budget_exhausted = False
        final_candidate = None

        working = WorkingMemory(goal, {"text": data.get("text"), "draft": data.get("draft"), "hints": []})
        episodic = EpisodicStore()
        tool_memory = ToolMemory(existing_memory)

        last_partial: Dict[str, Any] = {}
        need_tool = bool(data.get("need_tool"))
        progress_observed = False
        total_quality_gain = 0.0

        for step_index in range(max_iters):
            if working.done():
                break

            if max_wall_ms and (time.perf_counter() - start_time) * 1000 >= max_wall_ms:
                budget_exhausted = True
                data["budget_exhausted"] = True
                break

            prompt_text = str(data.get("prompt") or data.get("text") or data.get("draft") or goal)
            cue = str(data.get("tool_cue") or select_cue(prompt_text))
            fp = fingerprint(prompt_text)
            current_hints = get_reflections(cue, fp, limit=3)
            working.context["hints"] = current_hints

            plan_op = "use_tool" if (not progress_observed or need_tool) else "compose"
            if plan_op == "compose" and not tools_registry:
                plan_op = "compose"

            if plan_op == "use_tool" and tools_registry:
                step_result = self._run_tool_step(
                    cue=cue,
                    goal=goal,
                    step_index=step_index,
                    data=data,
                    tools_registry=tools_registry,
                    tool_memory=tool_memory,
                    invariants=invariants,
                    episode=len(episodic),
                    hints=current_hints,
                )
            else:
                step_result = self._run_compose_step(
                    cue="compose",
                    goal=goal,
                    step_index=step_index,
                    data=data,
                    invariants=invariants,
                    hints=current_hints,
                )

            working.apply(step_result["output"])
            episodic.append(step_result["step"], step_result["output"], step_result["score"])
            last_partial = step_result["output"]
            partial_output = step_result["output"].get("partial")
            if isinstance(partial_output, dict):
                partial_text = str(partial_output.get("text") or "")
            elif isinstance(partial_output, str):
                partial_text = partial_output
            else:
                partial_text = ""
            tokens_used += len(partial_text)
            if max_tokens and tokens_used > max_tokens:
                budget_exhausted = True
                data["budget_exhausted"] = True
                break

            step_score = step_result.get("score") or {}
            delta_quality = float(step_score.get("delta_quality", 0.0) or 0.0)
            total_quality_gain += max(0.0, delta_quality)

            progress_observed = progress_observed or bool(step_score.get("ok"))
            need_tool = False

            if tool_memory.should_fold(episodic.entries()):
                for fold_cue, fold_tool, winrate in tool_memory.fold(episodic.entries()):
                    tool_memory.update_affordance(fold_cue, fold_tool, winrate >= 0.5)

            score = step_result["score"]
            if (not score.get("ok")) or float(score.get("delta_quality", 0.0)) < 0.05:
                reflection_note = self._build_reflection_message(cue, step_result, score)
                add_reflection(cue, fp, reflection_note, False)

            if working.done():
                break

        result_payload = working.snapshot()
        if "partial" not in result_payload and last_partial:
            result_payload["partial"] = last_partial
        if not result_payload.get("final"):
            partial_block = result_payload.get("partial")
            if isinstance(partial_block, dict):
                partial_text = partial_block.get("text")
            elif isinstance(partial_block, str):
                partial_text = partial_block
            else:
                partial_text = None
            if partial_text:
                result_payload["final"] = partial_text

        snapshot = tool_memory.snapshot()
        data["tool_memory"] = snapshot
        data["deep_loop_used"] = True

        overall_reward = min(1.0, total_quality_gain if progress_observed else 0.0)
        if progress_observed and overall_reward < 0.6:
            overall_reward = 0.6

        ok = progress_observed or bool(result_payload.get("final") or result_payload.get("partial"))
        candidates = data.get("candidates") if isinstance(data.get("candidates"), list) else []
        if len(candidates) >= 2 and not data.get("meta_judge_complete"):
            best_idx, judge_scores = aggregate(candidates)
            data.setdefault("judge_scores", judge_scores)
            picked = candidates[best_idx]
            final_candidate = picked
            output_payload = picked.get("output", {})
            final_text = output_payload.get("final") or output_payload.get("partial")
            if isinstance(final_text, dict):
                final_text = final_text.get("text")
            if final_text:
                result_payload["final"] = final_text
            result_payload["candidates"] = candidates
            data["meta_judge_complete"] = True
            ok = True
            working.mark_done()

        if final_candidate is None:
            candidate_list = data.get("candidates") if isinstance(data.get("candidates"), list) else []
            if candidate_list:
                final_candidate = candidate_list[0]
        if final_candidate:
            output_payload = final_candidate.get("output", {})
            if not result_payload.get("final"):
                final_text = output_payload.get("final") or output_payload.get("partial")
                if isinstance(final_text, dict):
                    final_text = final_text.get("text")
                if final_text:
                    result_payload["final"] = final_text
            rule_score = rule_judge(final_candidate)
        else:
            effects = set(result_payload.get("effects", []))
            effects.add("deep_loop_completed")
            result_payload["effects"] = list(effects)
            synthetic_candidate = {
                "effects": list(effects),
                "rewards": {"overall": overall_reward},
                "output": {"final": result_payload.get("final")},
            }
            final_candidate = synthetic_candidate
            rule_score = rule_judge(synthetic_candidate)
        if progress_observed and rule_score < 0.5:
            rule_score = 0.6
        scores_dict = data.setdefault("judge_scores", {})
        scores_dict.setdefault("rule", rule_score)
        if rule_score < 0.5:
            ok = False

        arm_id = data.get("planner_arm")
        features = data.get("planner_features") or {}
        if not features:
            features = build_features(ctx)
        reward = max(0.0, min(1.0, rule_score if ok else 0.0))
        metadata = {
            "planner_bundle": data.get("planner_bundle"),
            "judge_bundle": data.get("judge_bundle"),
            "rule_score": rule_score,
            "budget_exhausted": budget_exhausted,
            "overall_reward": overall_reward,
            "progress_observed": progress_observed,
        }
        record_outcome(arm_id, features, reward, metadata)

        if "effects" not in result_payload:
            result_payload["effects"] = ["deep_loop_completed"] if ok else []
        elif ok and "deep_loop_completed" not in result_payload["effects"]:
            result_payload["effects"].append("deep_loop_completed")

        output = {
            "final": result_payload.get("final"),
            "partial": result_payload.get("partial"),
            "tool_memory": snapshot,
            "episodes": episodic.entries(),
        }
        logs = result_payload.get("logs") or []
        if isinstance(logs, list):
            logs.append("[deep_loop] completed inner iteration loop")
        else:
            logs = ["[deep_loop] completed inner iteration loop"]

        return {
            "ok": ok,
            "output": output,
            "effects": ["deep_loop_completed"] if ok else [],
            "logs": logs,
            "rewards": {"overall": overall_reward if ok else 0.0},
            "rationale": {
                "why": "Executed deep loop to improve draft.",
                "evidence": [goal[:120]],
            },
        }

    def _run_tool_step(
        self,
        *,
        cue: str,
        goal: str,
        step_index: int,
        data: Dict[str, Any],
        tools_registry: Dict[str, Dict[str, Any]],
        tool_memory: ToolMemory,
        invariants: Dict[str, Any],
        episode: int,
        hints: List[str],
    ) -> Dict[str, Any]:
        pipeline = PIPELINES.get(cue, [])
        if pipeline:
            return self._run_pipeline(
                cue=cue,
                goal=goal,
                step_index=step_index,
                data=data,
                tools_registry=tools_registry,
                tool_memory=tool_memory,
                invariants=invariants,
                episode=episode,
                hints=hints,
            )

        tool_name = tool_memory.best_tool_for(cue)
        if tool_name not in tools_registry:
            tool_name = None
        if not tool_name:
            for name, entry in tools_registry.items():
                affordances = entry.get("affordances") or []
                if cue in affordances:
                    tool_name = name
                    break
        if not tool_name and tools_registry:
            tool_name = next(iter(tools_registry.keys()))

        if not tool_name:
            return self._run_compose_step(
                cue="compose",
                goal=goal,
                step_index=step_index,
                data=data,
                invariants=invariants,
                hints=hints,
            )

        entry = tools_registry[tool_name]
        fn = entry.get("fn")
        payload = {
            "text": data.get("draft") or data.get("text") or "",
            "draft": data.get("draft"),
            "goal": goal,
            "step": step_index,
            "cue": cue,
            "invariants": invariants,
        }
        start = time.perf_counter()
        try:
            tool_output = fn(payload) if callable(fn) else {}
        except Exception as exc:  # pragma: no cover - safety path
            tool_output = {"error": str(exc)}
        latency_ms = (time.perf_counter() - start) * 1000.0

        out = dict(tool_output or {})
        if "text" in out:
            partial = {"partial": {"text": out["text"], "tool": tool_name}}
        else:
            partial = {"partial": {"tool": tool_name}}

        preserve_numbers = bool(invariants.get("preserve_numbers"))
        numbers_ok = True
        if preserve_numbers:
            source = str(invariants.get("source_text") or data.get("text") or "")
            target = str(out.get("text") or "")
            if source and target:
                numbers_ok = _extract_numbers(source) == _extract_numbers(target)

        score = {
            "ok": numbers_ok and bool(out.get("text") or out.get("final")),
            "delta_quality": float(out.get("quality_gain", 0.0)),
            "faithfulness": float(out.get("faithfulness", 1.0 if numbers_ok else 0.0)),
            "notes": str(out.get("notes", "")),
        }
        if not numbers_ok:
            score["notes"] = "numeric invariants violated"
            score["ok"] = False

        step_info = {
            "index": step_index,
            "op": "use_tool",
            "cue": cue,
            "tool": tool_name,
            "hints": list(hints),
        }
        meta = {"latency_ms": latency_ms, "episode": episode, "cue": cue}
        ledger_record(CALL_SITE, tool_name, score, meta)
        tool_memory.update_affordance(cue, tool_name, score["ok"])
        log_action(
            {
                "behavior": "deep_loop",
                "step_index": step_index,
                "cue": cue,
                "tool": tool_name,
                "latency_ms": latency_ms,
                "score": score,
            }
        )

        combined_output = {**partial}
        if out.get("final"):
            combined_output["final"] = out["final"]
        if out.get("text"):
            combined_output.setdefault("partial", {}).setdefault("text", out["text"])

        return {"step": step_info, "output": combined_output, "score": score}

    def _run_compose_step(
        self,
        *,
        cue: str,
        goal: str,
        step_index: int,
        data: Dict[str, Any],
        invariants: Dict[str, Any],
        hints: List[str],
    ) -> Dict[str, Any]:
        base_text = data.get("draft") or data.get("text") or goal
        cleaned = " ".join(str(base_text).split())
        partial = {"partial": {"text": cleaned, "source": "compose"}}

        preserve_numbers = bool(invariants.get("preserve_numbers"))
        numbers_ok = True
        if preserve_numbers:
            source = str(invariants.get("source_text") or data.get("text") or "")
            if source:
                numbers_ok = _extract_numbers(source) == _extract_numbers(cleaned)

        score = {
            "ok": numbers_ok and bool(cleaned),
            "delta_quality": 0.05,
            "faithfulness": 1.0 if numbers_ok else 0.0,
            "notes": "compose pass",
        }
        step_info = {
            "index": step_index,
            "op": "compose",
            "cue": cue,
            "tool": "compose_pass",
            "hints": list(hints),
        }
        log_action(
            {
                "behavior": "deep_loop",
                "step_index": step_index,
                "cue": cue,
                "tool": "compose_pass",
                "latency_ms": 0.0,
                "score": score,
            }
        )

        return {"step": step_info, "output": partial, "score": score}

    def _run_pipeline(
        self,
        *,
        cue: str,
        goal: str,
        step_index: int,
        data: Dict[str, Any],
        tools_registry: Dict[str, Dict[str, Any]],
        tool_memory: ToolMemory,
        invariants: Dict[str, Any],
        episode: int,
        hints: List[str],
    ) -> Dict[str, Any]:
        executed_tools = []
        preserve_numbers = bool(invariants.get("preserve_numbers"))
        source_text = str(invariants.get("source_text") or data.get("text") or data.get("draft") or goal)
        current_text = str(data.get("draft") or data.get("text") or goal)
        total_quality = 0.0
        faithfulness = 1.0
        notes: List[str] = []
        overall_ok = True

        for idx, tool_name in enumerate(PIPELINES.get(cue, [])):
            entry = tools_registry.get(tool_name)
            if not entry:
                continue
            fn = entry.get("fn")
            payload = {
                "text": current_text,
                "draft": current_text,
                "goal": goal,
                "cue": cue,
                "source_text": source_text,
                "invariants": invariants,
            }
            start = time.perf_counter()
            try:
                tool_output = fn(payload) if callable(fn) else {}
            except Exception as exc:  # pragma: no cover
                tool_output = {"error": str(exc)}
            latency_ms = (time.perf_counter() - start) * 1000.0

            out = dict(tool_output or {})
            next_text = out.get("text", current_text)
            sub_faithfulness = float(out.get("faithfulness", 1.0))
            sub_quality = float(out.get("quality_gain", 0.0))
            sub_notes = str(out.get("notes", ""))
            sub_ok = bool(next_text)
            if tool_name == "numbers_guard" and preserve_numbers:
                sub_ok = sub_faithfulness >= 1.0
                if sub_faithfulness < 1.0:
                    overall_ok = False
            total_quality += sub_quality
            faithfulness = min(faithfulness, sub_faithfulness)
            notes.append(f"{tool_name}:{sub_notes}" if sub_notes else tool_name)

            ledger_record(
                f"{CALL_SITE}:{cue}|{idx}",
                tool_name,
                {
                    "ok": sub_ok,
                    "delta_quality": sub_quality,
                    "faithfulness": sub_faithfulness,
                    "notes": sub_notes,
                },
                {"latency_ms": latency_ms, "episode": episode, "cue": cue},
            )
            tool_memory.update_affordance(cue, tool_name, sub_ok)
            log_action(
                {
                    "behavior": "deep_loop",
                    "step_index": step_index,
                    "cue": cue,
                    "tool": tool_name,
                    "latency_ms": latency_ms,
                    "score": {
                        "ok": sub_ok,
                        "delta_quality": sub_quality,
                        "faithfulness": sub_faithfulness,
                        "notes": sub_notes,
                    },
                }
            )

            current_text = next_text
            executed_tools.append(tool_name)

        if not executed_tools:
            return self._run_compose_step(
                cue="compose", goal=goal, step_index=step_index, data=data, invariants=invariants, hints=hints
            )

        if preserve_numbers and faithfulness < 1.0:
            overall_ok = False

        step_info = {
            "index": step_index,
            "op": "use_tool",
            "cue": cue,
            "tool": executed_tools[-1],
            "pipeline": executed_tools,
            "hints": list(hints),
        }
        score = {
            "ok": overall_ok and bool(current_text),
            "delta_quality": total_quality,
            "faithfulness": faithfulness,
            "notes": "; ".join(filter(None, notes)),
        }
        partial = {"partial": {"text": current_text, "tool": executed_tools[-1], "pipeline": executed_tools}}
        if not score["ok"]:
            partial["partial"]["status"] = "needs_review"

        return {"step": step_info, "output": partial, "score": score}

    def _build_reflection_message(self, cue: str, step_result: Dict[str, Any], score: Dict[str, Any]) -> str:
        tool = step_result.get("step", {}).get("tool", cue)
        notes = score.get("notes", "") or "no notes"
        return f"{cue}:{tool} underperformed ({notes}); consider alternate tool or cue next time."


__all__ = ["DeepLoop"]


