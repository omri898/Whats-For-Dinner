# What's for Dinner?

A multi-agent dinner advisor CLI where three AI friends with distinct personalities debate what you should cook tonight. Built with PydanticAI and a local vLLM model (Qwen3-14B).

## What it does

You pick your laziness level, a cuisine preference, and required ingredients from your pantry. Three agents — **Chef Enthusiastico** (the enthusiast), **The Lazy Advisor** (effort gatekeeper), and **Dr. Nutricia** (nutrition evangelist) — argue in a group chat until they agree on a recipe. This happens 3 times in sequence, with each round targeting a distinct recipe so you end up with 3 genuinely different recommendations.

## Setup

Requires a running vLLM server with `qwen3-14b`:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-14B-AWQ \
    --served-model-name qwen3-14b \
    --port 8001 \
    --tool-call-parser qwen3_xml \
    --enable-auto-tool-choice \
    --reasoning-parser qwen3
```

Install dependencies (in the `dinner` conda environment):

```bash
pip install "pydantic-ai[openai]" rich httpx
```

## Usage

```bash
make run          # interactive CLI
make run --debug  # show thinking tokens + raw context
```

Or test individual phases:

```bash
make check-vllm        # verify vLLM is running
make test-chef         # Chef agent alone
make test-round        # all three agents, one round
make test-discussion   # 3 sequential rounds + recommendations
make notebook          # open the developer notebook
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

Three PydanticAI agents share a flat message history (group chat pattern) within each round. Each agent has a `@system_prompt` decorator that serializes only the context fields relevant to its role — Chef sees the full pantry, Lazy sees the laziness level (privately), and Nutricia only sees proposed recipe ingredients.

The discussion runs as **3 sequential rounds**, each targeting one recipe. A round exits immediately when Lazy and Nutricia both approve the same dish. Agreed recipe names are passed forward between rounds so Chef is instructed to propose something different each time — different dish type, different primary protein. If a round hits `MAX_TURNS` without agreement, the recipe with the most approvals is picked as a fallback.

Built as a portfolio project demonstrating multi-agent coordination with structured output, personality-driven prompts, and a local inference backend.
