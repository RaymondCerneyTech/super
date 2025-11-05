from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

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

        profile_name = str(payload.get("profile") or payload.get("llama_profile") or data.get("llama_profile") or "")
        if not profile_name:
            profile_name = ""

        variables_raw = payload.get("vars") or payload.get("variables") or data.get("llama_vars") or {}
        variables = _normalize_vars(variables_raw)

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

        try:
            result = llama_runner.run_inference(
                prompt=str(prompt),
                model=model_path,
                n_predict=n_predict,
                temperature=temperature,
                extra_args=extra_args,
            )
        except llama_runner.LlamaBinaryNotFound as exc:
            return {"ok": False, "logs": [f"llama_generate: {exc}"]}

        if result["returncode"] != "0":
            logs = [f"llama_generate: llama exited with status {result['returncode']}"]
            if result.get("stderr"):
                logs.append(result["stderr"])
            return {
                "ok": False,
                "logs": logs,
                "output": {"stderr": result.get("stderr", ""), "command": result.get("command", "")},
            }

        output_text = result.get("stdout", "")
        payload_output = {
            "text": output_text,
            "stderr": result.get("stderr", ""),
            "command": result.get("command", ""),
            "model": str(model_path),
            "profile": profile_name or None,
        }
        data["llama_output"] = payload_output

        logs = ["llama_generate completed"]
        if profile_name:
            logs.append(f"profile={profile_name}")

        return {
            "ok": True,
            "output": payload_output,
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


__all__ = ["LlamaGenerate"]

