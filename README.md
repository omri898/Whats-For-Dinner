# What's for Dinner?

A multi-agent dinner advisor CLI where three AI friends with distinct personalities debate what you should cook tonight.

## What it does

You pick a laziness level, a cuisine preference, and ingredients from your pantry. Three agents — **Chef Enthusiastico** (recipe finder), **The Lazy Advisor** (effort gatekeeper), and **Dr. Nutricia** (nutrition evangelist) — argue in a group chat until they agree on a dish. This runs 3 sequential rounds, each producing a distinct recipe recommendation.

Chef searches for real recipes via MCP tools. Nutricia looks up actual nutritional data. Lazy makes sure you won't regret the effort.

## Setup

### Prerequisites

- Python >= 3.11
- A running vLLM server (Qwen3.5-27B-GPTQ-Int4)
- Node.js (for MCP tool servers)

### Install

```bash
git clone <this-repo>
cd multi-agent-dinner
pip install -e .
```

### MCP tool servers

Chef uses [`recipe-mcp-server`](https://www.npmjs.com/package/recipe-mcp-server):

```bash
npm install -g recipe-mcp-server
```

Nutricia uses `mcp-opennutrition` (nutrition data lookups):

```bash
git clone https://github.com/deadletterq/mcp-opennutrition
```

See `skills/nutricia-mcp.md` for setup details.

### Start the vLLM server

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

## Usage

```bash
make run          # interactive CLI
make run-debug    # show thinking tokens + raw context
```

Test individual phases:

```bash
make check-vllm       # verify vLLM is running
make test-chef        # Chef agent alone
make test-round       # all three agents, one round
make test-discussion  # 3 sequential rounds + recommendations
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
