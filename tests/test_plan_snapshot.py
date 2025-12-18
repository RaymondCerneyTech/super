import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_cli(tmp_env: dict[str, str], *args: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(REPO_ROOT / "main.py"), *args]
    env = os.environ.copy()
    env.update(tmp_env)
    return subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)


def test_plan_snapshot_records_sources(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    env = {"SUPER_INDEX_DIR": str(tmp_path / "indexes")}

    doc1 = tmp_path / "doc1.txt"
    doc1.write_text(
        "The Riemann Hypothesis strategy memo emphasizes analytic number theory tactics and risk controls.",
        encoding="utf-8",
    )
    doc2 = tmp_path / "doc2.txt"
    doc2.write_text(
        "Collaboration notes describe governance updates and shared resources for complex conjectures.",
        encoding="utf-8",
    )

    run_cli(env, "ingest", "--path", str(doc1))
    run_cli(env, "ingest", "--path", str(doc2))

    rollouts_path = Path(".ai") / "rollouts.jsonl"
    count_before = 0
    if rollouts_path.exists():
        count_before = len(rollouts_path.read_text(encoding="utf-8").splitlines())

    question = "Summarize the latest Riemann Hypothesis research tactics and cite sources."
    result = run_cli(
        env,
        "plan",
        "--goal",
        "llm_output,formatted,grounded,cited",
        "--text",
        question,
        "--skip-llama",
        "--max-expansions",
        "6",
        "--no-bandit",
    )
    output = result.stdout
    assert "Sources:" in output
    assert "doc1.txt" in output or "doc2.txt" in output

    assert rollouts_path.exists()
    count_after = len(rollouts_path.read_text(encoding="utf-8").splitlines())
    assert count_after >= count_before + 4

    snapshot_path = Path(".ai") / "last_plan.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["index_backend"] == "hnsw"
    assert snapshot["sources"]
    assert any("doc1.txt" in src.get("title", "") for src in snapshot["sources"])
    assert isinstance(snapshot.get("react_trace"), list) and snapshot["react_trace"]
    sc_votes = snapshot.get("sc_votes", snapshot["data"].get("sc_votes"))
    auto_query = snapshot.get("auto_query", snapshot["data"].get("auto_query"))
    assert isinstance(sc_votes, list)
    assert isinstance(auto_query, list)
    assert snapshot.get("mpc_mode") in {"ensemble", "heuristic"}
    assert isinstance(snapshot.get("web_trace"), list)
    top_passages = snapshot.get("top_passages") or []
    assert top_passages
    assert all("title" in entry for entry in top_passages[:1])
    assert snapshot.get("bundle_arm")
    assert isinstance(snapshot.get("planner_bundle"), list)
    assert isinstance(snapshot.get("judge_bundle"), list)
    assert snapshot.get("model_bundle") is not None
