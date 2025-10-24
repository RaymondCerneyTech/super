API Server
-----------
Run the FastAPI wrapper to expose `/ask` and `/plan` endpoints:

```bash
uvicorn server:app --reload --port 8000
```

POST to `/ask` with a JSON payload such as `{"question": "Summarize ..."}` (plus optional overrides) to receive an answer and the plan metadata.
You can pass `config_path` in the payload to reuse CLI configs.

Act with `--task`
-----------------
The CLI can now run simple natural-language automation commands. It defaults to a dry-run preview (read-only); pass `--approve` and the necessary `--permit` flags to execute.

```
# Preview
python main.py --task "Download https://example.com/file.zip to downloads/file.zip and unzip downloads/file.zip to data/"

# Approve with write/net permissions
python main.py --task "Download https://example.com/file.zip to downloads/file.zip and unzip downloads/file.zip to data/" --approve --permit read,write,net
```

🧠 Super AI — Scriptable Unified Process for Evolved Reasoning

Author: Ray Cerney (2025)
Core idea: Super AI is a modular interpreter + adapter framework that treats cognition as an evolving network of behaviors, not static neural weights.

🌍 Overview

Super AI defines intelligence as:

Intelligence = Inference + Preference
• Inference → patterns → meaning
• Preference → patterns from patterns → selection of what to do next

Instead of large opaque models, Super AI builds reasoning from small, scriptable behaviors that can be created, combined, or evolved independently.
Each behavior is a self-contained process with inputs, outputs, and metadata describing when or why it should run.
A lightweight interpreter routes data and control between behaviors.

This structure mirrors how biological and collective systems (brains, Reddit, Twitch, Stack Overflow) form distributed intelligences: many local inferences, one evolving preference space.

⚙️ Architecture
/super/
│
├── core/
│   ├── interpreter.py     # Executes and routes behaviors
│   ├── router.py          # Decides which behavior to trigger next
│   └── memory.py          # Optional: stores behavior states/preferences
│
├── behaviors/
│   ├── __init__.py
│   └── example_behavior.py  # Template for new behaviors
│
├── tests/
│   └── test_basic.py
│
└── main.py                # Entry point / orchestration script

🧩 Behavior Model

Each behavior follows a simple interface:

class Behavior:
    def __init__(self, name: str, inputs: list[str], outputs: list[str]):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs

    def run(self, context):
        """Perform inference or preference transformation."""
        raise NotImplementedError


Behaviors can call or spawn others, forming a behavior graph.
The interpreter manages flow, logging, and adaptation logic.

🧬 Goals for Codex / Contributors

Codex or any collaborator should focus on:

Interpreter layer — execute behaviors in sequence or as a DAG.

Behavior registry — discover, load, and describe available behaviors.

Preference module — evolve which behaviors are chosen (basic RL or heuristic).

Persistence — save learned configurations or new behaviors to disk.

Examples — build demo behaviors (math solver, text rewriter, file reader).

🚀 Future Extensions

Behavior evolution via reinforcement learning or genetic search.

Natural-language “behavior scripts.”

Integration with human input (Super AI ↔ Mega AI collective loop).

Self-Improving Router (LinUCB) & Logs + Plan Cache
-------------------------------------

The router now learns whether analytic or creative behaviors perform better for similar requests using a lightweight LinUCB contextual bandit. Each `plan` run appends one JSON Lines record to `logs/bandit.jsonl`, capturing the goal, extracted features, chosen cluster, rewards, and any unmet goal flags. JSON Lines is newline-delimited JSON, ideal for tailing or streaming analysis.

Inspect learning trends:

```
python tools/analyze_bandit.py --log-file logs/bandit.jsonl --window 50
```

Add `--csv-out path/to/report.csv` to export rows for external plotting. Use `--no-bandit` on the CLI to disable bandit updates while still logging for comparisons.

Web Ingestion + Verbose Answers
-----------------------------------

Quickly pull external knowledge into the local index and generate cited, sectioned answers directly from the CLI.

Ingest a single page:

```
python main.py ingest --url https://example.com/guide --tags "guide,example" --index-backend hnsw
```

Ingest a feed:
The HTML extractor keeps only the central article/main content and filters navigation or cookie banners; for best results, tag sources for precise retrieval.


```
python main.py ingest --rss https://example.com/feed.xml --index-backend hnsw
```

Ask for a long, cited answer:

```
python main.py ask --question "Explain X in depth with examples" --cited --grounded --verbose --min-words 1200 --fresh 30 --index-backend hnsw --explain
```

Answers borrow the `answer_verbose` behavior, which coordinates retrieval, optional aggregation, verification, and citation formatting for grounded responses. Add `--fresh DAYS` to bias retrieval toward recently ingested material when timeliness matters.

Why Super AI Instead of ChatGPT
--------------------------------
- Deterministic workflow orchestration: you design the pipeline (retrieve → check → format → verify) and each step is logged and auditable.
- Full control of data: ingest feeds, local files, or codebases and tag them for precision; nothing is hidden in a remote training snapshot.
- Automation friendly: plan caching, router learning, and behavior metadata make repeat tasks fast, reusable, and consistent.
- Extensible by design: drop new behaviors into `behaviors/` and wire them into plans without retraining a large model.

Make It Friendlier & General Use
--------------------------------
Create a config file (defaults live in `config/examples/super.yaml`) and run commands with `--config super.yaml` or set `SUPER_CONFIG_PATH`. Example:

```yaml
ingest:
  index_backend: hnsw
  tags: quickstart

ask:
  k_passages: 10
  max_chars: 9000
  min_words: 900
  cited: true
  grounded: true
  fresh_days: 30

plan:
  max_expansions: 18
```

```bash
python main.py --config super.yaml ask --question "Summarize our latest policy updates" --explain
```

- **Bundle starter kits:** ship ingest seeds and ready-made plans for common domains (compliance, research, coding) to shorten setup time.
- **Expose simple toggles:** support CLI/`super.yaml` configs plus env vars so non-developers can adjust backends, caching, and logging.
- **Offer a quick UI:** a lightweight FastAPI or Streamlit app to ingest docs, run plans, and inspect answers/logs.
- **Instrument analytics:** pair the plan/bandit analyzers with dashboards to monitor cache hit rates, failures, and recency coverage.
- **Document behaviors:** keep `.meta.yaml` files descriptive—consider a “behavior gallery” folder so others can plug in modules.
- **Provide an API wrapper:** optional REST endpoints so external systems can trigger `ask`/`plan` workflows programmatically.

🪶 Licensing

Code: MIT License © 2025 Ray Cerney

Documentation / Theory: Creative Commons Attribution 4.0 International (CC BY 4.0)

Credit: “Super AI – Scriptable Unified Process for Evolved Reasoning by Ray Cerney.”
Full CC license text - https://creativecommons.org/licenses/by/4.0/

## Behavior-Based Reasoning: Planner + Bandit

The CLI orchestrates behaviors with a best-first planner guided by a lightweight contextual bandit. Behaviors publish effects (have_summary, compliant, formatted) and multi-metric rewards; the planner composes them until the requested goal flags are satisfied.

Example (1 minute):

```
python main.py plan --goal "summary,compliant,formatted" \
  --text "Draft contract includes the secret roadmap." \
  --explain
```

Sample trace:

```
Plan: summarize -> policy_check -> document_formatting
1. summarize
   why: Selected summary using extractive strategy
   evidence: Artificial intelligence..., It enables automation...
   effects: have_summary
   rewards: overall=0.88 {"brevity": 1.0, "relevance": 0.75}
2. policy_check
   why: Flagged and redacted prohibited phrases
   evidence: secret roadmap
   effects: compliant
   rewards: overall=0.95 {"manageable": 1.0, "preservation": 0.9}
3. document_formatting
   why: Formatted document in business style
   evidence: headings: ..., tone: ..., wrap=80
   effects: formatted
   rewards: overall=0.92 {"readability": 0.9, "preservation": 0.85}
Final reward:
{
  "brevity": 1.0,
  "relevance": 0.75,
  "fluency": 0.8,
  "explanation_presence": 1.0,
  "overall": 0.93
}
```

Tweak the goal flags or text to explore other behaviour combinations; the router will gradually bias between analytic and creative clusters based on observed rewards.
