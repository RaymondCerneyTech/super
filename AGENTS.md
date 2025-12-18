# AGENTS

## Orchestration Layers

* **Entrypoints**
  * `main.py` - CLI hub for `summarize`, `format`, `optimize`, `check`, `ingest`, `ask`, `plan`, and automation-focused `do` tasks. It now accepts `--skip-llama` for offline runs and emits plan snapshots to `.ai/last_plan.json`.
  * `server.py` - FastAPI wrapper exposing `/ask` and `/plan` HTTP endpoints on top of the planner/runtime stack.

* **Planner Layer**
  * `core/planner.py` - best-first / GOAP-style planner with goal-flag utility shaping, deep-loop fallback, world-model (MPC) hooks, and code-edit pipeline scoring.
  * `core/plans.py` - static YAML/JSON plan templates consumed by the interpreter.
  * `core/plan_cache.py` - persistent cache keyed by goal flags, backend, verbosity, tags, and meaning.
  * `planners/registry.py` - registry for First-Principles, analogy, causal, ReAct, and Tree-of-Thought planners via `run_planner(name, ctx)`.

* **Router & Bandit Layer**
  * `core/router.py` - LinUCB contextual bandit, feature extractor, and fallback logic across analytic/creative clusters (meaning-aware).
  * `core/learn.py` - adapter/bias learner for historical reward shaping.
  * `core/rewards.py` - aggregation helpers (overall, citation coverage, faithfulness, verbosity) shared across layers.
  * `tools/analyze_bandit.py` - offline analyzer for JSONL bandit logs (moving averages, per-cluster statistics, optional CSV export).

* **Runtime Layer**
  * `core/interpreter.py` - executes behaviors, enforces preconditions, records rewards/checks.
  * `core/audit.py`, `core/logs.py` - JSONL logging helpers and audit sink for CLI/automation traces; the planner also snapshots final plans into `.ai/last_plan.json`.
  * `core/memory.py`, `core/reflections.py`, `core/credit_ledger.py`, `core/checks.py`, `core/features.py`, `core/sandbox.py` - inner-loop working memory, reflections store, per-tool credit ledger, validation metrics, feature extractors, and sandboxed FS helpers.

* **Behaviors**
  * Retrieval & knowledge: `behaviors/retrieve.py`, `behaviors/aggregate.py`, `behaviors/ingest_web.py`, `behaviors/report_from_data.py`.
  * Writing & editing: `behaviors/summarize.py`, `behaviors/answer_verbose.py`, `behaviors/document_formatting.py`, `behaviors/rewrite_style.py`.
  * Governance & analysis: `behaviors/policy_check.py`, `behaviors/grammar_correction.py`, `behaviors/sentiment_analysis.py`, `behaviors/social_post_optimize.py`.
  * Automation primitives: `behaviors/files_read.py`, `behaviors/files_write.py`, `behaviors/command_parse.py`, plus `.meta.yaml` descriptors.
  * Planning orchestration: `behaviors/meta_pipeline.py` delegates to the meta-planner to emit 1-N candidate plans (ReAct / ToT / First-Principles blends).
  * Tool-assisted reasoning: `behaviors/deep_loop.py` runs DeepAgent-style inner loops, calling `tools/registry.py` entries, honoring invariants, and logging to `.ai/ledger_tool_calls.jsonl`.
  * LLM generation: `behaviors/llama_generate.py` wraps llama.cpp invocations with profile support or short-circuits when the context signals `skip_llama`.

## Code Editing Behavior

- `behaviors/refactor_code.py`: Refactors targeted functions into `async` stubs when requests call for async upgrades.
- `behaviors/code_edit.py`: Orchestrates code-edit tasks, applying import fixes or delegating to specialized helpers based on the user prompt.
- `behaviors/add_endpoint.py`: Generates FastAPI endpoint scaffolds (router + Pydantic models) for requests such as "add a user login endpoint."
- Code-generation runs append rich audit entries to `logs/codegen.jsonl` (request text, chosen branch, effects, previews) so pipelines can be replayed or analyzed later. Returned logs include `[codegen] action=... status=...` traces for each step.
- `behaviors/deep_loop.py`: DeepAgent-style inner loop that samples tools from `tools/registry.py`, maintains working memory and episodic traces, updates tool credit, and records reflections when chains underperform.

## Tool Power Pack

- Web utilities (`tools/web.py`): `web_get`, `html_to_text`, `extract_links`, `extract_facts` (10 s timeout, 200 kB cap).
- Data helpers (`tools/data_utils.py`): `table_detect`, `csv_summary`, `json_query`, `dedupe_lines`.
- Math & units (`tools/math_units.py`): `math_eval_safe`, `unit_convert_basic`.
- Planning aids (`tools/planning.py`): `causal_dot_builder`, `timeline_normalize`.
- Memory & cache (`tools/memory_tools.py`): `reflection_add`, `reflection_get`, `cache_put`, `cache_get`.
- Repo utilities (`tools/aci.py`): `grep_repo`, `read_file`, `write_file`, `append_file`, `run_pytest` (last 40 lines). Legacy aliases (`code_read`, `code_write`, `search_repo`, `run_tests`) remain available.
- All tools register affordances in `tools/registry.py`. Cue defaults (`core/cues.py`) now map `web` -> `["web_get","html_to_text","extract_facts","numbers_guard"]` and `data` -> `["table_detect","csv_summary"]`.
- Validation: `tests/test_tools_powerpack.py` exercises web, data, and repo flows (`python -m pytest tests/test_tools_powerpack.py -q`).

## Run & Setup Commands

```bash
# FastAPI server
uvicorn server:app --reload --port 8000

# Automation preview & execution
python main.py --task "Download https://example.com/file.zip to downloads/file.zip and unzip downloads/file.zip to data/"
python main.py --task "Download https://example.com/file.zip to downloads/file.zip and unzip downloads/file.zip to data/" --approve --permit read,write,net

# Knowledge ingestion & retrieval
python main.py ingest --url https://www.claymath.org/millennium-problems/riemann-hypothesis/ --tags "math,riemann" --index-backend hnsw
python main.py ingest --rss https://feeds.feedburner.com/QuantaMagazineMathPhysics --limit 5 --index-backend hnsw
python main.py ask --question "Explain current strategies for the Riemann Hypothesis" --cited --grounded --verbose --min-words 600 --index-backend hnsw --skip-llama --explain

# Planner walkthrough
python main.py plan --goal "llm_output,grounded,cited,formatted" --text "Analyze current tactics for the Riemann Hypothesis." --llama-profile research_plan --skip-llama --max-expansions 8 --explain

# Config-driven invocation
python main.py --config super.yaml ask --question "Summarize our latest policy updates" --explain
```

## LLM & World-Model Controls

- Use `--skip-llama` on `plan`, `ask`, or `do` when llama.cpp is unavailable; planners still run retrieval, aggregation, verification, and formatting behaviors, and outputs are snapshotted to `.ai/last_plan.json`.
- After capturing rollouts, refresh the learned dynamics with `python scripts/train_dynamics.py --rollouts .ai/rollouts.jsonl --output .ai/dynamics_ensemble.pt --steps 500`; subsequent plans automatically load the ensemble for `mpc_plan`.
- Inspect recent planner decisions, rewards, and answers via `.ai/last_plan.json`, or replay `.ai/rollouts.jsonl` alongside `tools/analyze_bandit.py`.

## Bandit Analyzer

```bash
python tools/analyze_bandit.py --log-file logs/bandit.jsonl --window 50
```

Add `--csv-out path/to/report.csv` for spreadsheet pipelines.

## Meaning-first Routing (historical reference)

The repository already contains:

- `behaviors/meaning_infer.py`
- Planner/router/audit updates that propagate `ctx["data"]["meaning"]`
- Cache keys and logs that include meaning

When editing the repo in the future:

1. **Do not recreate** `behaviors/meaning_infer.py`.
2. **Check** `core/planner.py`, `core/router.py`, `core/interpreter.py`, `core/audit.py`, and `core/plan_cache.py` continue threading `data["meaning"]`.
3. Run `python main.py plan --goal "summary,formatted" --text "test" --explain` (or `python -m pytest`) after meaning-related edits.
