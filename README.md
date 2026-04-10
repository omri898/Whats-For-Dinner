# What's for Dinner?

A multi-agent dinner advisor CLI where three AI friends with distinct personalities debate what you should cook tonight. Built with PydanticAI, MCP tool servers, and a local vLLM model (Qwen3.5-27B-GPTQ-Int4).

## What it does

You pick your laziness level, a cuisine preference, and required ingredients from your pantry. Three agents — **Chef Enthusiastico** (the enthusiast), **The Lazy Advisor** (effort gatekeeper), and **Dr. Nutricia** (nutrition evangelist) — argue in a group chat until they agree on a recipe. This happens 3 times in sequence, with each round targeting a distinct recipe so you end up with 3 genuinely different recommendations.

Chef searches for real recipes via MCP tools, Nutricia looks up actual nutritional data, and Lazy makes sure you won't regret the effort.

## Setup

### Prerequisites

- Python >= 3.11
- A running vLLM server with Qwen3.5-27B
- Node.js (for MCP tool servers)

### Install dependencies

```bash
conda activate dinner
pip install -e .
```

Or manually:

```bash
pip install "pydantic-ai[openai,mcp]" rich httpx python-dotenv
```

### Start the vLLM server

```bash
make start-vllm-2-4090   # 2x 4090 GPUs
make start-vllm-6000     # single GPU on port 8001
```

Or manually:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-27B-GPTQ-Int4 \
    --served-model-name qwen3.5-27b \
    --port 8001 \
    --quantization gptq_marlin \
    --tool-call-parser qwen3_xml \
    --enable-auto-tool-choice \
    --reasoning-parser qwen3
```

### MCP tool servers

Chef uses [`recipe-mcp-server`](https://www.npmjs.com/package/recipe-mcp-server) for recipe search:

```bash
npm install -g recipe-mcp-server
make check-mcp   # verify it's installed
```

Nutricia uses `mcp-opennutrition` (included as a submodule in `mcp-opennutrition/`) for nutritional data lookups. See `skills/nutricia-mcp.md` for setup details.

```bash
make check-nutrition-mcp   # verify it's configured
```

## Usage

```bash
make run          # interactive CLI
make run-debug    # show thinking tokens + raw context
```

Test individual phases:

```bash
make check-vllm          # verify vLLM is running
make test-chef            # Chef agent alone
make test-round           # all three agents, one round
make test-discussion      # 3 sequential rounds + recommendations
make notebook             # open the developer notebook
```

## Example output

```
╭─ Chef Enthusiastico ─────────────────────────────────╮
│ Shakshuka! Eggs poached in spiced tomato — one pan,   │
│ done in 20 minutes, and yes, feta on top.             │
╰──────────── Turn 1 · proposal · → all · Shakshuka ───╯

╭─ The Lazy Advisor ────────────────────────────────────╮
│ Chef. One pan? Fine. But if I have to julienne         │
│ anything, I'm ordering pizza.                          │
╰──────────── Turn 2 · reaction · → chef · Shakshuka ──╯

╭─ Dr. Nutricia ────────────────────────────────────────╮
│ Lycopene from the tomatoes, protein from the eggs,     │
│ and the feta adds calcium. I'll allow it.              │
╰──────────── Turn 3 · reaction · → all · Shakshuka ───╯
```

## Architecture

```
src/
├── main.py         # CLI entry point — interactive prompts via Rich
├── agents.py       # PydanticAI agents, MCP tool setup, schema sanitisation
├── discussion.py   # Group chat orchestration, 3-round logic, Rich UI
├── models.py       # Pydantic data classes (GroupMessage, GroupContext, RecipeCard)
└── config.py       # Model name, temperatures, max tokens, vLLM endpoint
```

Three PydanticAI agents share a flat message history (group chat pattern) within each round. Each agent has a `@system_prompt` decorator that serializes only the context fields relevant to its role — Chef sees the full pantry, Lazy sees the laziness level (privately), and Nutricia only sees proposed recipe ingredients.

### MCP tools

Agents connect to external data through the [Model Context Protocol](https://modelcontextprotocol.io/):

- **Chef** uses `recipe_search` and `recipe_get` to find real recipes matching the cuisine and ingredients
- **Nutricia** uses `search-food-by-name` to look up actual nutritional info for proposed dishes

Tool visibility is scoped per agent via `MCPServerStdio.filtered()` — each agent only sees the tools relevant to its role.

### Discussion flow

The discussion runs as **3 sequential rounds**, each targeting one recipe. A round exits immediately when Lazy and Nutricia both approve the same dish. Agreed recipe names are passed forward between rounds so Chef is instructed to propose something different each time — different dish type, different primary protein. If a round hits `MAX_TURNS` without agreement, the recipe with the most approvals is picked as a fallback.

### vLLM compatibility

Several workarounds handle Qwen3.5-27B quirks under vLLM:

- **`_StripReasoningTransport`** — custom HTTP transport that strips the non-standard `reasoning` field injected by vLLM's `--reasoning-parser qwen3`
- **Schema sanitisation** — removes `anyOf/oneOf` with null and `default` keys that are incompatible with vLLM's `qwen3_xml` tool-call parser
- **Tool content truncation** — JSON-aware truncation to prevent parser errors on oversized MCP responses

## Project structure

```
multi-agent-dinner/
├── src/                    # application code
├── mcp-opennutrition/      # nutrition MCP server (submodule)
├── data/ingredients.json   # pantry ingredients list
├── scripts/                # phase-test scripts
├── skills/                 # developer skill docs
├── rules/                  # developer workflow docs
├── logs/                   # generated at runtime
├── Makefile
├── pyproject.toml
└── CLAUDE.md
```

---

Built as a portfolio project demonstrating multi-agent coordination with structured output, personality-driven prompts, MCP tool integration, and a local inference backend.
