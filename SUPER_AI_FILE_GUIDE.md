# Super AI – Developer Guide (File Roles & How to Use)

This guide explains what each module in your current upload does and how to run it from the CLI or API. It’s based on the code present in the uploaded files under `/mnt/data` as of this export.

---

## Top-level entry points

### `main.py`
**Role:** Command‑line entrypoint. Wires up the behavior registry, interpreter, router, and planner; logs runs; loads config; and exposes a suite of subcommands.

**Key commands (excerpted from built‑in help):**
- `summarize --text TEXT [--strategy {trim,extractive,abstractive}] [--max-words N] [--max-sentences N]`  
- `format --text TEXT [--style {business,casual}] [--wrap-width N]`
- `optimize --text TEXT [--platform P] [--topic T] [--max-length N]`
- `check --text TEXT --policies "phrase1,phrase2"`
- `ingest --url URL | --rss FEED | --path FILE [--tags "t1,t2"] [--index-backend {tfidf,hnsw}] [--limit N]`
- `ask --question "..." [--k-passages N] [--max-chars N] [--tags TAGS] [--index-backend BACKEND] [--cited] [--grounded] [--verbose] [--min-words N] [--max-words N] [--log-file PATH] [--no-bandit] [--explain]`
- `plan --goal "summary,compliant,formatted" [--text TEXT] [--log-file PATH] [--no-bandit] [--explain] [--max-expansions N]`
- `do --task "natural language task" [--workspace DIR] [--permit read,write,net,exec] [--approve]`

**Quick examples:**
```bash
# 1) Plan a multi-step behavior sequence with trace
python main.py plan --goal "summary,compliant,formatted" --text "Draft contract includes the secret roadmap." --explain

# 2) Ingest a webpage to the local index and ask a grounded, cited question
python main.py ingest --url "https://example.com/guide" --tags "guide,example" --index-backend hnsw
python main.py ask --question "Explain transformers" --cited --grounded --verbose --min-words 1200 --explain

# 3) Safe file operations from natural language (dry run by default; add --approve to execute)
python main.py do --task "Download https://example.com/file.zip to downloads/file.zip and unzip downloads/file.zip to data/"
python main.py do --task "..." --approve --permit read,write,net
```

---

### `server.py`
**Role:** FastAPI server exposing a minimal HTTP surface.

**Endpoints:**
- `GET /health` → health check.
- `POST /ask` → request body includes `question`, optional retrieval/verbosity options; returns `answer`, `sources`, `plan`, `rewards`, `goal_satisfied`, `expansions`.
- `POST /plan` → request body includes `goal`, optional `text` and other fields; returns the constructed plan.

Run with `uvicorn server:app --reload` (or the equivalent command in your environment).

---

## Planning, routing, and execution

### `planner.py`
**Role:** Goal‑to‑plan search. Normalizes human goals (e.g., `"summary"`, `"cited"`, `"formatted"`) via `GOAL_ALIASES` into effect flags, explores behavior sequences, respects preconditions, merges reward estimates, and caches successful plans for reuse.

**Use:** Primarily invoked via `python main.py plan ...` or the `/plan` API. You can tune max expansions and enable `--explain` to emit trace details.

### `router.py`
**Role:** Heuristic behavior selector. Extracts features from the current goal and content (conciseness, compliance hints, etc.), consults the registry, and orders candidate behaviors; includes fallback when a chosen behavior under‑performs.

**Use:** Implicit—used by the planner/interpreter to prefer the next action.

### `interpreter.py`
**Role:** Executes a single behavior robustly. Validates required inputs, runs the behavior, writes outputs back into `ctx["data"]`, computes checks, aggregates rewards, and applies an explanation bonus.

**Use:** Implicit—called by the CLI/server when running a plan or a single behavior step.

---

## Retrieval & answering pipeline (RAG-like)

### `ingest_web.py` – Behavior `ingest_web`
**Role:** Fetch a URL, RSS feed, or local file and add its content to the local index; attach optional `tags`; choose index backend (`tfidf` or `hnsw`).

**Inputs:** `url` or `rss` or `path`, optional `tags`, `index_backend`.  
**Outputs:** (side‑effect) adds to index.  
**Run via CLI:** `python main.py ingest --url ... --tags ... --index-backend hnsw`

### `retrieve.py` – Behavior `retrieve`
**Role:** Given a `question` (and optional `tags`), pull top‑K passages from the chosen index backend.  
**Inputs:** `question`, optional `tags`, `k_passages`, `max_chars`, `index_backend`.  
**Outputs:** `passages` (list of dicts).

### `aggregate.py` – Behavior `aggregate`
**Role:** Combine retrieved passages into `aggregated_text` using a chosen strategy (e.g., simple concatenation or topic grouping).  
**Inputs:** `passages`, optional `aggregate_strategy`.  
**Outputs:** `aggregated_text`.

### `answer_verbose.py` – Behavior `answer_verbose`
**Role:** Produce a long‑form answer with optional citations/faithfulness/verbosity rewards. Respects `min_words`/`max_words` and verbosity modes.  
**Inputs:** `question`, `passages` and/or `aggregated_text`, `verbosity`, `min_words`, `max_words`.  
**Outputs:** `answer`.  
**Tip:** Pair with `retrieve` → `aggregate` in plans for cited, grounded answers.

---

## Editing & analysis behaviors

### `summarize.py` – Behavior `summarize`
**Role:** Summarize `text` using one of: `trim`, `extractive`, or `abstractive`.  
**Inputs:** `text`, strategy/length limits.  
**Outputs:** `summary`.

### `document_formatting.py` – Behavior `document_formatting`
**Role:** Normalize headings, bullets, spacing; optional `style` (e.g., business/casual).  
**Inputs:** `text`, optional `format_style`, `wrap_width`.  
**Outputs:** `formatted_text`.

### `rewrite_style.py` – Behavior `rewrite_style`
**Role:** Adjust tone/voice (e.g., friendlier, more creative). Returns clear failure if the requested tone can’t be satisfied.  
**Inputs:** `text`, `tone`/style hints.  
**Outputs:** tone‑adjusted text and effect flags like `tone_adjusted`, `creative_tone`.

### `grammar_correction.py` – Behavior `grammar_correction`
**Role:** Fix basic grammar/punctuation issues.  
**Inputs:** `text`.  
**Outputs:** `corrected_text`.

### `sentiment_analysis.py` – Behavior `sentiment_analysis`
**Role:** Very lightweight sentiment tagger using simple lexicons.  
**Inputs:** `text`.  
**Outputs:** `sentiment` (label + supporting words).

### `policy_check.py` – Behavior `policy_check`
**Role:** Scan `text` for prohibited phrases/policies; emit violations and a redacted/sanitized version.  
**Inputs:** `text`, optional `policies`.  
**Outputs:** `violations`, `sanitized_text`.

---

## Task parsing & safe file operations

### `command_parse.py` – Behavior `command_parse`
**Role:** Parse a natural‑language `task` into concrete steps (download, unzip, move/copy/delete, etc.).  
**Inputs:** `task`.  
**Outputs:** `steps` (structured actions for the executor).

### `files_move_copy_delete.py` – Behavior `files_move_copy_delete`
**Role:** Execute file operations in a permission‑gated sandbox.  
**Inputs:** `op` (`move|copy|delete`), `src`, optional `dst`, flags `recursive`, and context keys `perms` (set), `dry_run` (bool).  
**Outputs:** none (logs + rewards).  
**Safety:** Dry‑run by default; requires `--approve` and explicit `--permit` flags in the CLI to actually mutate files.

### `sandbox.py`
**Role:** The permissioned execution environment used by file operations and shell‑like steps. Supports a command allow‑list (`zip`, `unzip`, `tar`, `python`, `node`), hashing, and audit logging.  
**Permissions:** `read`, `write`, `net`, `exec` (grants are checked per operation).

---

## Logs & docs

### `runs.log`, `actions.log`
**Role:** Plaintext/JSONL‑style logs produced by CLI/server runs and sandbox actions (good inputs for a simple bandit analyzer).

### `README.md`
**Role:** Landing instructions and high‑level overview. This guide adds concrete file‑level roles and usage.

---

## Common workflows

1) **RAG answer with citations**
```bash
python main.py ingest --url "https://example.com/tutorial" --tags "tutorial"
python main.py ask --question "How do transformers work?" --cited --grounded --verbose --min-words 800 --explain
```

2) **Summarize → format → policy‑check (as a plan)**
```bash
python main.py plan --goal "summary,formatted,compliant" --text "Your long draft..." --explain
```

3) **Natural‑language task → parsed steps → safe file ops**
```bash
python main.py do --task "Download https://example.com/archive.zip to downloads/archive.zip and unzip to data/"
# If the dry-run looks correct:
python main.py do --task "Download https://example.com/archive.zip to downloads/archive.zip and unzip to data/" --approve --permit read,write,net,exec
```

---

## Configuration hints

- Place a `super.yaml` next to `main.py` (or pass `--config`) to set defaults like retrieval backend (`tfidf`/`hnsw`), `ask.k_passages`, `plan.max_expansions`, and logging paths.
- Use `--explain` on `plan`/`ask` to surface router choices, plan cache hits, rewards, and step‑by‑step traces.

---

If you want, I can add a starter `super.yaml` and a tiny FastAPI HTML page that hits `/ask` and `/plan` next.
