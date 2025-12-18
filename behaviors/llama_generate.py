from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

from core.interfaces import Behavior, Context, Result
from core import llama_profiles, models as model_store
from tools import llama_runner


class LlamaGenerate(Behavior):
    name = "llama_generate"
    inputs: List[str] = []
    outputs: List[str] = ["llama_output"]

    def run(self, ctx: Context) -> Result:
        data = ctx.setdefault("data", {})
        payload = data.get("llama")
        if not isinstance(payload, Mapping):
            payload = {}

        profile_name = str(
            payload.get("profile")
            or payload.get("llama_profile")
            or data.get("llama_profile")
            or ""
        ).strip()
        variables_raw = payload.get("vars") or payload.get("variables") or data.get("llama_vars") or {}
        variables = _normalize_vars(variables_raw)

        sources, source_passages = _prepare_sources(data)
        citable_sources = [src for src in sources if src.get("url")]
        aggregated_text = str(data.get("aggregated_text") or "")
        if not aggregated_text and source_passages:
            aggregated_text = _fallback_body_from_sources(citable_sources or sources, source_passages)
            if aggregated_text:
                data["aggregated_text"] = aggregated_text

        question = _resolve_question(ctx, data)
        requires_grounded = _needs_grounded_output(data, profile_name)
        if requires_grounded and not citable_sources:
            return {
                "ok": False,
                "logs": ["llama_generate: no citable sources available for grounded output"],
                "effects": [],
                "reward": 0.0,
            }

        if data.get("skip_llama"):
            fallback_body = aggregated_text or question or data.get("text") or ctx.get("text") or ""
            fallback_body = str(fallback_body)
            final_text = _format_grounded_answer(fallback_body, citable_sources or sources, source_passages)
            payload_output = {
                "text": final_text,
                "stderr": "",
                "command": "",
                "model": "skipped",
                "profile": profile_name or None,
            }
            data["llama_output"] = payload_output
            data["answer"] = final_text
            data["text"] = final_text
            data["_llm_generated"] = True
            data.setdefault("sc_votes", [])
            logs = ["llama_generate skipped (requested)"]
            return {
                "ok": True,
                "output": payload_output,
                "logs": logs,
                "effects": ["llm_output"],
            }

        prompt = (
            payload.get("prompt")
            or data.get("llama_prompt")
            or ctx.get("text")
            or data.get("text")
            or ""
        )

        profile_settings: Optional[Dict[str, object]] = None
        if profile_name:
            try:
                profile_settings = llama_profiles.resolve_profile(profile_name, variables)
            except KeyError as exc:
                return {
                    "ok": False,
                    "logs": [f"llama_generate: {exc}"],
                }

        if not prompt and profile_settings:
            prompt = str(profile_settings.get("prompt", "") or "")
        if prompt and variables and not profile_settings:
            prompt = llama_profiles.render_template(str(prompt), variables)

        apply_research_prompt = bool(data.get("apply_research_prompt")) or requires_grounded
        if profile_name == "research_plan" and apply_research_prompt and (citable_sources or sources):
            prompt = _build_research_prompt(question, aggregated_text, citable_sources or sources, source_passages)

        if not prompt:
            return {"ok": False, "logs": ["llama_generate: missing prompt."]}

        model_arg = (
            payload.get("model")
            or (profile_settings.get("model") if profile_settings else None)
            or data.get("llama_model")
        )
        try:
            model_path = model_store.resolve_model_path(str(model_arg) if model_arg else None)
        except FileNotFoundError as exc:
            return {"ok": False, "logs": [f"llama_generate: {exc}"]}

        n_predict = _pick_number(payload.get("n_predict"), profile_settings, "n_predict")
        temperature = _pick_number(payload.get("temperature"), profile_settings, "temperature")

        extra_args: List[str] = []
        if profile_settings:
            extras = profile_settings.get("extra")
            if isinstance(extras, list):
                extra_args.extend(str(item) for item in extras)
        extra_override = payload.get("extra")
        if isinstance(extra_override, list):
            extra_args.extend(str(item) for item in extra_override)
        elif isinstance(extra_override, str):
            extra_args.append(extra_override)

        data["llama_prompt"] = prompt
        sc_samples = int(data.get("sc_samples") or data.get("self_consistency") or data.get("sc") or 1)
        samples: List[Dict[str, Any]] = []
        generation_errors: List[str] = []

        def _run_once(temp_override: Optional[float]) -> Dict[str, Any]:
            try:
                return llama_runner.run_inference(
                    prompt=str(prompt),
                    model=model_path,
                    n_predict=n_predict,
                    temperature=temp_override if temp_override is not None else temperature,
                    extra_args=extra_args,
                )
            except llama_runner.LlamaBinaryNotFound as exc:  # pragma: no cover
                raise RuntimeError(str(exc))

        vote_table: List[Dict[str, Any]] = []
        kept_markers: Set[str] = {src.get("marker") for src in (citable_sources or sources) if src.get("marker")}
        try:
            if sc_samples > 1 and not data.get("skip_llama"):
                sc_outputs: List[str] = []
                for _ in range(sc_samples):
                    sample_temp = temperature if temperature is not None else 0.6
                    sample = _run_once(sample_temp)
                    if sample.get("returncode") != "0":
                        generation_errors.append(sample.get("stderr", ""))
                        continue
                    text_out = str(sample.get("stdout", "") or "")
                    sc_outputs.append(text_out)
                    samples.append(sample)
                if not sc_outputs:
                    raise RuntimeError("all self-consistency samples failed")
                final_text_raw, vote_table, kept_markers = _self_consistency_vote(sc_outputs, sources)
                data["sc_votes"] = vote_table
                result = samples[vote_table[0]["index"]] if samples else _run_once(temperature)
                raw_output = final_text_raw
            else:
                result = _run_once(temperature)
                if result.get("returncode") != "0":
                    raise RuntimeError(result.get("stderr", "llama inference failed"))
                raw_output = str(result.get("stdout", "") or "")
                data["sc_votes"] = vote_table or [
                    {"index": 0, "text": raw_output.strip(), "markers": list(kept_markers)}
                ]
        except RuntimeError as exc:
            error_logs = [f"llama_generate: {exc}"]
            if generation_errors:
                error_logs.extend(generation_errors)
            return {"ok": False, "logs": error_logs}

        filtered_sources = [
            src for src in (citable_sources or sources) if src.get("marker") in kept_markers
        ]
        if not filtered_sources:
            filtered_sources = list(citable_sources or sources)

        cleaned_body = raw_output.strip()
        if not cleaned_body:
            cleaned_body = _fallback_body_from_sources(filtered_sources, source_passages)
        final_text = _format_grounded_answer(cleaned_body, filtered_sources, source_passages)
        if requires_grounded and not filtered_sources:
            return {
                "ok": False,
                "logs": ["llama_generate: unable to ground answer after self-consistency voting"],
                "effects": [],
                "reward": 0.0,
            }

        payload_output = {
            "text": final_text,
            "stderr": (result.get("stderr") if isinstance(result, dict) else ""),
            "command": (result.get("command") if isinstance(result, dict) else ""),
            "model": str(model_path),
            "profile": profile_name or None,
        }
        data["llama_output"] = payload_output
        data["answer"] = final_text
        data["text"] = final_text
        data["_llm_generated"] = True
        data["sources"] = filtered_sources
        data["has_citable_sources"] = bool(filtered_sources)

        logs = ["llama_generate completed"]
        if profile_name:
            logs.append(f"profile={profile_name}")
        if citable_sources:
            logs.append(f"sources={len(citable_sources)}")

        return {"ok": True,            "output": payload_output,
            "logs": logs,
            "effects": ["llm_output"],
        }


def _normalize_vars(raw: Any) -> Dict[str, str]:
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    if isinstance(raw, list):
        result: Dict[str, str] = {}
        for item in raw:
            if isinstance(item, str) and "=" in item:
                key, value = item.split("=", 1)
                result[key.strip()] = value
        return result
    if isinstance(raw, str) and "=" in raw:
        key, value = raw.split("=", 1)
        return {key.strip(): value}
    return {}


def _pick_number(value: Any, profile: Optional[Dict[str, object]], key: str) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if profile and isinstance(profile.get(key), (int, float)):
        return float(profile[key])  # type: ignore[index]
    return None


def _prepare_sources(data: Dict[str, Any]) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
    sources = _normalize_source_list(data.get("sources"))
    raw_passages = data.get("source_passages")
    if isinstance(raw_passages, dict):
        source_passages = {str(k): str(v or "") for k, v in raw_passages.items()}
    else:
        source_passages = {}

    if not sources and isinstance(data.get("passages"), list):
        built_sources, built_map = _build_sources_from_passages(data.get("passages") or [])
        if built_sources:
            data["sources"] = built_sources
            data["source_passages"] = built_map
            sources = built_sources
            source_passages = built_map

    return sources, source_passages


def _normalize_source_list(raw: Any) -> List[Dict[str, str]]:
    if not isinstance(raw, list):
        return []
    normalized: List[Dict[str, str]] = []
    seen_markers: set[str] = set()
    for entry in raw:
        if not isinstance(entry, Mapping):
            continue
        marker = str(entry.get("marker") or "").strip()
        if not marker:
            marker = f"S{len(normalized) + 1}"
        if marker in seen_markers:
            continue
        seen_markers.add(marker)
        title = str(entry.get("title") or entry.get("source") or f"Source {len(normalized) + 1}").strip()
        url = str(entry.get("url") or "").strip()
        if url and not url.lower().startswith(("http://", "https://", "file://")):
            url = ""
        normalized.append({"marker": marker, "title": title, "url": url})
    return normalized


def _build_sources_from_passages(passages: List[Mapping[str, Any]]) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
    sources: List[Dict[str, str]] = []
    marker_to_text: Dict[str, str] = {}
    seen_keys: set[Tuple[str, str]] = set()
    for passage in passages:
        if not isinstance(passage, Mapping):
            continue
        meta = passage.get("meta")
        if not isinstance(meta, Mapping):
            meta = {}
        title = str(
            meta.get("title")
            or meta.get("url")
            or meta.get("canonical_url")
            or meta.get("source_id")
            or meta.get("path")
            or f"Source {len(sources) + 1}"
        ).strip()
        raw_url = meta.get("canonical_url") or meta.get("url") or meta.get("path") or ""
        url = str(raw_url).strip() if isinstance(raw_url, str) else ""
        if (not url or not isinstance(raw_url, str)) and meta.get("path"):
            try:
                url = Path(str(meta["path"]).strip()).resolve().as_uri()
            except Exception:
                url = ""
        if url and not url.lower().startswith(("http://", "https://", "file://")):
            url = ""
        key = (title, url)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        marker = f"S{len(sources) + 1}"
        sources.append({"marker": marker, "title": title or marker, "url": url})
        marker_to_text[marker] = str(passage.get("text", ""))
    return sources, marker_to_text


def _fallback_body_from_sources(sources: List[Dict[str, str]], source_passages: Mapping[str, str]) -> str:
    if not sources:
        return ""
    lines = ["## Key Findings"]
    for src in sources:
        marker = src.get("marker") or ""
        snippet = _summarize_for_prompt(source_passages.get(marker, ""))
        if snippet:
            lines.append(f"- {snippet} [{marker}]")
    return "\n".join(lines).strip()


def _resolve_question(ctx: Context, data: Mapping[str, Any]) -> str:
    candidates = (
        data.get("question"),
        ctx.get("question"),
        data.get("text"),
        ctx.get("text"),
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return ""


def _needs_grounded_output(data: Mapping[str, Any], profile_name: str) -> bool:
    goal_flags = data.get("goal_flags")
    flag_set = set()
    if isinstance(goal_flags, (list, tuple, set)):
        flag_set = {str(flag).lower() for flag in goal_flags}
    grounded_requested = bool(flag_set.intersection({"grounded", "cited"}))
    if bool(data.get("grounded")) or bool(data.get("cited")):
        grounded_requested = True
    if bool(data.get("allow_ungrounded_profile")) and not grounded_requested:
        return False
    if profile_name == "research_plan":
        return True
    return grounded_requested


def _summarize_for_prompt(text: str, max_chars: int = 500) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    truncated = cleaned[:max_chars].rsplit(" ", 1)[0].strip()
    return truncated + "..."


def _build_research_prompt(
    question: str,
    aggregated_text: str,
    sources: List[Dict[str, str]],
    source_passages: Mapping[str, str],
) -> str:
    question_line = question.strip() or "Provide an update based on the context."
    context_sections: List[str] = []
    if aggregated_text.strip():
        context_sections.append("Aggregated Notes:\n" + aggregated_text.strip())
    if sources:
        lines: List[str] = []
        for src in sources:
            marker = src.get("marker") or ""
            title = src.get("title") or marker
            url = src.get("url") or ""
            header = f"[{marker}] {title}"
            if url:
                header += f" ({url})"
            lines.append(header)
            snippet = _summarize_for_prompt(source_passages.get(marker, ""))
            if snippet:
                lines.append(snippet)
        context_sections.append("Source Passages:\n" + "\n".join(lines))
    context_block = "\n\n".join(section for section in context_sections if section).strip()
    instructions = (
        "You are preparing a grounded research memo.\n"
        "Use only the context below. Do NOT invent facts or sources.\n"
        "- Structure the memo with clear headings (e.g., Overview, Opportunities, Risks, Next Steps).\n"
        "- Attribute factual statements with inline citations using the provided markers (e.g., [S1]).\n"
        "- Highlight disagreements or uncertainties when present.\n"
        "- Finish with a section titled \"Sources:\" listing each cited source as [S#] Title - URL.\n"
    )
    prompt_parts = [
        instructions.strip(),
        f"Question: {question_line}",
    ]
    if context_block:
        prompt_parts.append("Context:\n" + context_block)
    return "\n\n".join(prompt_parts).strip()


def _remove_existing_sources_section(text: str) -> str:
    lines = str(text or "").splitlines()
    trimmed: List[str] = []
    dropping = False
    for line in lines:
        if not dropping and line.strip().lower().startswith("sources:"):
            dropping = True
            continue
        if dropping:
            continue
        trimmed.append(line)
    return "\n".join(trimmed).strip()


def _force_inline_tags(text: str, sources: List[Dict[str, str]]) -> str:
    if not sources:
        return text.strip()
    body = text.strip()
    missing = [src for src in sources if src.get("marker") and f"[{src['marker']}]" not in body]
    if not missing:
        return body
    paragraphs = [p for p in body.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [body]
    for idx, src in enumerate(missing):
        marker = src.get("marker") or ""
        target_idx = min(idx, len(paragraphs) - 1)
        paragraphs[target_idx] = paragraphs[target_idx].rstrip() + f" [{marker}]"
    return "\n\n".join(paragraphs).strip()


def _ensure_sources_section(text: str, sources: List[Dict[str, str]]) -> str:
    body = text.strip()
    if not sources:
        return body
    sources_with_urls = [src for src in sources if src.get("url")]
    if not sources_with_urls:
        return body
    lines = [body, "", "Sources:"]
    for src in sources_with_urls:
        lines.append(f"  [{src['marker']}] {src['title']} - {src['url']}")
    return "\n".join(lines).strip()


def _format_grounded_answer(body: str, sources: List[Dict[str, str]], source_passages: Mapping[str, str]) -> str:
    cleaned = _remove_existing_sources_section(body)
    if not cleaned.strip():
        cleaned = _fallback_body_from_sources(sources, source_passages)
    cleaned = _force_inline_tags(cleaned, sources)
    return _ensure_sources_section(cleaned, sources)


def _self_consistency_vote(
    outputs: List[str],
    sources: List[Dict[str, str]],
) -> Tuple[str, List[Dict[str, Any]], Set[str]]:
    if not outputs:
        markers = {src.get("marker") for src in sources if src.get("marker")}
        return "", [], markers

    majority = (len(outputs) + 1) // 2
    marker_pattern = re.compile(r"\[S\d+\]")
    segment_stats: Dict[str, Dict[str, Any]] = {}
    global_markers: Counter[str] = Counter()

    for raw in outputs:
        text = (raw or "").strip()
        lines = [line for line in text.splitlines() if line.strip()]
        seen_segments: Set[str] = set()
        for order, line in enumerate(lines):
            prefix_match = re.match(r"^(\s*[-*]\s*)", line)
            prefix = prefix_match.group(1) if prefix_match else ""
            content = marker_pattern.sub("", line).strip()
            if not content and not line.strip().startswith("#"):
                continue
            key = content.lower()
            entry = segment_stats.setdefault(
                key,
                {
                    "prefix": prefix,
                    "content": content,
                    "order": order,
                    "count": 0,
                    "marker_counts": Counter(),
                },
            )
            entry["order"] = min(entry["order"], order)
            if key not in seen_segments:
                entry["count"] += 1
                seen_segments.add(key)
            markers = marker_pattern.findall(line)
            for marker in markers:
                entry["marker_counts"][marker] += 1
                global_markers[marker] += 1

    kept_segments: List[Tuple[int, str]] = []
    kept_markers: Set[str] = {marker for marker, count in global_markers.items() if count >= majority}
    if not kept_markers:
        kept_markers = {src.get("marker") for src in sources if src.get("marker")}

    for entry in segment_stats.values():
        if entry["count"] < majority:
            continue
        prefix = entry["prefix"]
        content = entry["content"]
        marker_choices = [marker for marker, count in entry["marker_counts"].items() if count >= majority]
        marker_choices = [marker for marker in marker_choices if marker in kept_markers]
        line_text = f"{prefix}{content}" if prefix else content
        if marker_choices:
            line_text = f"{line_text} {' '.join(marker_choices)}".strip()
            kept_markers.update(marker_choices)
        elif not line_text.lstrip().startswith("#"):
            continue
        kept_segments.append((entry["order"], line_text))

    kept_segments.sort(key=lambda item: item[0])
    final_lines: List[str] = []
    for _, line in kept_segments:
        if line.strip().startswith("##") and final_lines:
            final_lines.append("")
        final_lines.append(line)

    final_text = "\n".join(final_lines).strip()
    if not final_text and outputs:
        final_text = outputs[0].strip()

    vote_summary: List[Dict[str, Any]] = []
    for key, entry in sorted(segment_stats.items(), key=lambda item: item[1]["order"]):
        vote_summary.append(
            {
                "text": entry["content"],
                "count": entry["count"],
                "markers": dict(entry["marker_counts"]),
            }
        )

    return final_text, vote_summary, kept_markers


__all__ = ["LlamaGenerate"]
