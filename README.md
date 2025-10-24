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

🪶 Licensing

Code: MIT License © 2025 Ray Cerney

Documentation / Theory: Creative Commons Attribution 4.0 International (CC BY 4.0)

Credit: “Super AI – Scriptable Unified Process for Evolved Reasoning by Ray Cerney.”
Full CC license text - https://creativecommons.org/licenses/by/4.0/
