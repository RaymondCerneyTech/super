# AGENTS

## Orchestration Layers

* **Entrypoints**

  * `main.py` — CLI hub for `summarize`, `format`, `optimize`, `check`, `ingest`, `ask`, `plan`, and the automation-focused `do` command.
  * `server.py` — FastAPI wrapper that exposes `/ask` and `/plan` HTTP endpoints around the planner/runtime stack.

* **Planner Layer**

  * `core/planner.py` — best-first / GOAP-style planner with goal-flag utility shaping and command-parse sub-step execution.
  * `core/plans.py` — static / YAML- or JSON-driven plans that feed behaviors through the interpreter.
  * `core/plan_cache.py` — persistent cache keyed by goal flags, backend, verbosity, and tags.

* **Router & Bandit Layer**

  * `core/router.py` — LinUCB contextual bandit, feature extractor, and fallback logic for analytic/creative clusters.
  * `core/learn.py` — adapter/bias learner for historical reward shaping.
  * `core/rewards.py` — aggregation helpers (overall, citation coverage, faithfulness, verbosity) shared across layers.
  * `tools/analyze_bandit.py` — offline analyzer for JSONL bandit logs (moving averages, per-cluster stats).

* **Runtime Layer**

  * `core/interpreter.py` — executes behaviors, enforces preconditions, records rewards/checks.
  * `core/audit.py`, `core/logs.py` — JSONL logging helpers and audit sink for CLI/automation traces.
  * `core/checks.py`, `core/features.py`, `core/sandbox.py` — reusable validation metrics, router features, and sandboxed FS operations.

* **Behaviors**

  * Retrieval & knowledge: `behaviors/retrieve.py`, `behaviors/aggregate.py`, `behaviors/ingest_web.py`, `behaviors/report_from_data.py`.
  * Writing & editing: `behaviors/summarize.py`, `behaviors/answer_verbose.py`, `behaviors/document_formatting.py`, `behaviors/rewrite_style.py`.
  * Governance & analysis: `behaviors/policy_check.py`, `behaviors/grammar_correction.py`, `behaviors/sentiment_analysis.py`, `behaviors/social_post_optimize.py`.
  * Automation primitives: `behaviors/files_read.py`, `behaviors/files_write.py`, `behaviors/command_parse.py`, plus associated `.meta.yaml` descriptors.

## Code Editing Behavior
- `behaviors/refactor_code.py`: Refactors targeted functions into `async` stubs when requests call for async upgrades.
- `behaviors/code_edit.py`: Orchestrates code-edit tasks, applying import fixes or delegating to specialized helpers based on the user prompt.
- `behaviors/add_endpoint.py`: Generates FastAPI endpoint scaffolds (router + Pydantic models) for requests such as “add a user login endpoint.”

## Run & Setup Commands

```bash
# FastAPI server
uvicorn server:app --reload --port 8000

# Automation preview & execution
python main.py --task "Download https://example.com/file.zip to downloads/file.zip and unzip downloads/file.zip to data/"
python main.py --task "Download https://example.com/file.zip to downloads/file.zip and unzip downloads/file.zip to data/" --approve --permit read,write,net

# Knowledge ingestion & retrieval
python main.py ingest --url https://example.com/guide --tags "guide,example" --index-backend hnsw
python main.py ingest --rss https://example.com/feed.xml --index-backend hnsw
python main.py ask --question "Explain X in depth with examples" --cited --grounded --verbose --min-words 1200 --fresh 30 --index-backend hnsw --explain

# Planner walkthrough
python main.py plan --goal "summary,compliant,formatted" --text "Draft contract includes the secret roadmap." --explain

# Config-driven invocation
python main.py --config super.yaml ask --question "Summarize our latest policy updates" --explain
```

## Bandit Analyzer

```bash
python tools/analyze_bandit.py --log-file logs/bandit.jsonl --window 50
```

## Meaning-first Routing (to add)

This codebase already has a planner, a contextual router, and an interpreter. To make them choose the *right* behaviors for the user’s real intent, add a meaning-inference behavior and thread its output through planner → router → audit.

### 1. Create the meaning behavior

Create this file:

**`behaviors/meaning_infer.py`**

```python
from core.interfaces import Behavior, Context, Result

class MeaningInfer(Behavior):
    name = "meaning_infer"
    inputs = ["text"]
    outputs = ["meaning"]

    def run(self, ctx: Context) -> Result:
        data = ctx.setdefault("data", {})
        text = (data.get("text") or ctx.get("text") or "").lower()

        if any(k in text for k in ["summarize", "summary", "shorten", "tl;dr"]):
            meaning = "compress_to_essence"
        elif any(k in text for k in ["rewrite", "tone", "style", "friendlier", "rn study"]):
            meaning = "transform_style"
        elif any(k in text for k in ["cite", "source", "grounded", "where did you get"]):
            meaning = "ground_and_cite"
        elif any(k in text for k in ["analyze", "audit", "check this"]):
            meaning = "analyze_and_comment"
        elif any(k in text for k in ["plan", "steps", "run this", "task"]):
            meaning = "plan_and_execute"
        else:
            meaning = "compress_to_essence"

        data["meaning"] = meaning
        return Result.ok({"meaning": meaning}, effects=[f"meaning:{meaning}"])
```

Notes:

* keep the same import style and base classes as your other behaviors
* expose the meaning as both data and an effect so planners can match on it

### 2. Planner must prefer meaning-matched behaviors

Edit **`core/planner.py`** so that, after you assemble candidate behaviors and before final scoring, you read the meaning and give a small bonus to matching behaviors:

```python
meaning = ctx.get("data", {}).get("meaning")
for beh in candidates:
    score = compute_base_score(beh)
    if meaning and any(meaning in e for e in beh.effects):
        score += 0.2
    # keep existing scoring and cache logic
```

### 3. Router/bandit must learn per meaning

Edit **`core/router.py`** so the contextual bandit key includes meaning:

```python
features_key = extract_features(ctx)
meaning = ctx.get("data", {}).get("meaning", "none")
bandit_key = f"{features_key}::{meaning}"
```

Then pass `bandit_key` to whatever LinUCB / learn routine you already call. This lets the learner build separate stats for e.g. `long_text::compress_to_essence` vs `long_text::ground_and_cite`.

### 4. Interpreter / audit must log meaning

Edit **`core/interpreter.py`** and/or **`core/audit.py`** so log records include the meaning:

```python
record["meaning"] = ctx.get("data", {}).get("meaning")
```

This makes `tools/analyze_bandit.py` and future automations able to group by meaning.

### 5. Test / verify

Agents should try to run **at least one** of these after applying the edits:

```bash
python -m pytest
# or, if tests are light:
python main.py plan --goal "summary,formatted" --text "test text for meaning routing" --explain
```

If tests fail, show the diff for the planner/router/interpreter files.

---

## Meaning-first (installed)

The repository now already contains:

- `behaviors/meaning_infer.py`
- planner/routing updates to read `ctx["data"]["meaning"]`
- audit/interpreter logging of `"meaning"`

When editing this repo in the future:

1. **Do not recreate** `behaviors/meaning_infer.py` if it exists.
2. **Do** re-check these files for consistency:
   - `core/planner.py` — meaning bonus on behavior scoring
   - `core/router.py` — bandit/context key includes meaning
   - `core/interpreter.py` / `core/audit.py` — log meaning in JSONL
   - `core/plan_cache.py` — cache key includes meaning
3. Run:
   ```bash
   python main.py plan --goal "summary,formatted" --text "test" --explain
