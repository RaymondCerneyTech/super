import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from env import CodeWorld, Task
from tools import llama_runner


def run_episode(
    task: Task,
    llm_call: Callable[[str, Optional[Dict[str, object]]], Tuple[str, Dict[str, object]]],
) -> List[Dict[str, object]]:
    env = CodeWorld(task)
    trace: List[Dict[str, object]] = []
    last_obs: Optional[Dict[str, object]] = None
    try:
        for _ in range(task.max_steps):
            state_summary = env.summarize_state()
            action_code, adapter_mix = llm_call(state_summary, last_obs)
            obs, reward, done, info = env.step(action_code)
            trace.append(
                {
                    "task_name": task.name,
                    "step": env.step_count,
                    "state_summary": state_summary,
                    "action_code": action_code,
                    "obs": obs,
                    "reward": reward,
                    "done": done,
                    "adapter_mix": adapter_mix,
                }
            )
            last_obs = obs
            if done:
                break
    finally:
        env.cleanup()
    return trace


def save_trace(trace: List[Dict[str, object]], out_dir: str = "traces") -> Path:
    if not trace:
        raise ValueError("trace must contain at least one record")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    task_name = trace[0]["task_name"]
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    file_path = out_path / f"{task_name}__{timestamp}.jsonl"
    with file_path.open("w", encoding="utf-8") as handle:
        for record in trace:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return file_path


def make_llm_call(
    model_path: str,
    *,
    temperature: float = 0.2,
    n_predict: int = 512,
    extra_args: Optional[List[str]] = None,
) -> Callable[[str, Optional[Dict[str, object]]], Tuple[str, Dict[str, object]]]:
    extras = extra_args or ["--n-gpu-layers", "0"]

    def _caller(
        state_summary: str, last_obs: Optional[Dict[str, object]]
    ) -> Tuple[str, Dict[str, object]]:
        prompt = _build_prompt(state_summary, last_obs)
        result = llama_runner.run_inference(
            prompt=prompt,
            model=Path(model_path),
            n_predict=n_predict,
            temperature=temperature,
            extra_args=extras,
        )
        raw_output = result.get("stdout", "")
        action_code = _extract_code(raw_output).strip()
        looks_like_prompt = action_code.lower().startswith("you are a coding agent")
        if not action_code or looks_like_prompt:
            debug_dump = json.dumps(result, ensure_ascii=False)[:2000]
            print(f"[llm_call] invalid action_code; raw result snippet: {debug_dump}")
            action_code = ""
        if not action_code:
            action_code = "# Model produced no code.\n"
        adapter_mix = {
            "model": model_path,
            "temperature": temperature,
            "n_predict": n_predict,
        }
        return action_code, adapter_mix

    return _caller


def _build_prompt(state_summary: str, last_obs: Optional[Dict[str, object]]) -> str:
    sections = [
        "You are a coding agent that must emit the entire contents of solution.py.",
        "Return only Python code, optionally inside ```python fences.",
        f"State summary:\n{state_summary}",
    ]
    if last_obs:
        sections.append(
            "Previous test results: "
            f"passed={last_obs.get('passed')} failed={last_obs.get('failed')} total={last_obs.get('total')}"
        )
    sections.append("Update the solution to maximize passed tests.")
    return "\n\n".join(sections)


def _extract_code(output: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", output, re.DOTALL)
    if match:
        return match.group(1)
    return output


if __name__ == "__main__":
    example_task = Task(
        name="reverse_string",
        initial_solution="""
def solve(s: str) -> str:
    return s[::-1]
""".strip(),
        tests="""
import pytest
from solution import solve

@pytest.mark.parametrize("text", ["abc", "", "palindrome", "12345"])
def test_reverse(text):
    assert solve(text) == text[::-1]
""".strip(),
    )

    model_path = os.getenv(
        "LLAMA_MODEL_PATH",
        r"C:\AI\models\deepseek-coder-6.7b-instruct.Q5_K_M.gguf",
    )
    if Path(model_path).exists():
        llm_call = make_llm_call(
            model_path,
            extra_args=["--n-gpu-layers", "0"],
        )
    else:
        print(f"Warning: model {model_path} not found. Using pass-through proposer.")

        def llm_call(state_summary, last_obs):
            return example_task.initial_solution, {"adapter": "fallback"}

    episode_trace = run_episode(example_task, llm_call)
    path = save_trace(episode_trace)
    print(f"Saved trace to {path}")






