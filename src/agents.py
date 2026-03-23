from __future__ import annotations

import json
import httpx
from rich import print as rprint
from rich.table import Table

from pydantic_ai import Agent, RunContext  # type: ignore
from pydantic_ai.mcp import MCPServerStdio  # type: ignore
from pydantic_ai.models.openai import OpenAIChatModel  # type: ignore
from pydantic_ai.providers.openai import OpenAIProvider  # type: ignore

from src.config import MODEL_NAME, VLLM_BASE
from src.models import (
    GroupContext,
    GroupMessage,
    LazyGroupContext,
)

# ---------------------------------------------------------------------------
# Schema sanitiser
#
# vLLM's qwen3_xml tool-call parser rejects two non-standard constructs:
#   - anyOf/oneOf with null  (Pydantic's encoding of Optional[T])
#   - "default" keys in property defs  (not in OpenAI's JSON Schema subset)
# MCP tool schemas may contain either. Walk and normalise in-place.
# ---------------------------------------------------------------------------

def _sanitize_schema(schema: dict) -> dict:
    if "anyOf" in schema:
        non_null = [s for s in schema["anyOf"] if s.get("type") != "null"]
        if len(non_null) == 1:
            schema.pop("anyOf")
            schema.update(non_null[0])
    schema.pop("default", None)
    for value in schema.get("properties", {}).values():
        _sanitize_schema(value)
    if isinstance(schema.get("items"), dict):
        _sanitize_schema(schema["items"])
    return schema


# ---------------------------------------------------------------------------
# HTTP transport: strips 'reasoning' from outgoing assistant messages
#
# vLLM's --reasoning-parser qwen3 injects a non-standard 'reasoning' field
# into assistant messages. PydanticAI echoes it back on subsequent requests,
# causing vLLM to 500 on any multi-turn tool call flow. Stripping it here
# keeps --reasoning-parser qwen3 intact for debug ThinkingPart tokens.
# ---------------------------------------------------------------------------

class _StripReasoningTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self._inner = httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        try:
            body = json.loads(request.content)
            for msg in body.get("messages", []):
                msg.pop("reasoning", None)
                if msg.get("role") == "assistant" and msg.get("content") is None:
                    msg["content"] = ""
            for tool in body.get("tools", []):
                params = tool.get("function", {}).get("parameters")
                if isinstance(params, dict):
                    _sanitize_schema(params)
            new_content = json.dumps(body).encode()
            headers = dict(request.headers)
            headers["content-length"] = str(len(new_content))
            request = httpx.Request(
                method=request.method,
                url=request.url,
                headers=headers,
                content=new_content,
            )
        except Exception:
            pass  # non-JSON request — leave untouched
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self._inner.aclose()


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_model() -> OpenAIChatModel:
    http_client = httpx.AsyncClient(transport=_StripReasoningTransport())
    return OpenAIChatModel(
        MODEL_NAME,
        provider=OpenAIProvider(
            base_url=VLLM_BASE,
            api_key="dummy",
            http_client=http_client,
        ),
    )


# ---------------------------------------------------------------------------
# MCP server — recipe search + detail
# ---------------------------------------------------------------------------

_CHEF_TOOLS = {"recipe_search", "recipe_get"}
_mcp = MCPServerStdio("recipe-mcp-server", [])
_mcp_filtered = _mcp.filtered(lambda _ctx, tool: tool.name in _CHEF_TOOLS)


# ---------------------------------------------------------------------------
# vLLM health check
# ---------------------------------------------------------------------------

async def check_vllm() -> None:
    """Hit /v1/models and pretty-print. Raises ConnectionError on failure."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{VLLM_BASE}/models",
                headers={"Authorization": "Bearer dummy"},
            )
            resp.raise_for_status()
    except (httpx.ConnectError, httpx.HTTPStatusError) as exc:
        raise ConnectionError(
            f"vLLM not reachable at {VLLM_BASE} — is the server running?"
        ) from exc

    models = resp.json().get("data", [])
    table = Table(title="vLLM Models")
    table.add_column("ID", style="bold cyan")
    table.add_column("Object")
    for m in models:
        table.add_row(m.get("id", "?"), m.get("object", "?"))
    rprint(table)


# ---------------------------------------------------------------------------
# Helper: extract latest proposed ingredients from history
# ---------------------------------------------------------------------------

def _latest_proposed(history: list[GroupMessage]) -> list[str]:
    """Return proposed_ingredients from the most recent proposal/pivot/lock, or []."""
    for msg in reversed(history):
        if msg.message_type in ("proposal", "pivot") and msg.proposed_ingredients:
            return msg.proposed_ingredients
    return []


# ---------------------------------------------------------------------------
# Chef Enthusiastico
# ---------------------------------------------------------------------------

chef_agent: Agent[GroupContext, GroupMessage] = Agent(  # type: ignore
    make_model(),
    deps_type=GroupContext,
    output_type=GroupMessage,
    retries=3,
    toolsets=[_mcp_filtered],
)


@chef_agent.system_prompt
def chef_system_prompt(ctx: RunContext[GroupContext]) -> str:  # type: ignore
    d = ctx.deps
    history_text = "\n".join(
        f"[{m.agent}] {m.text}" for m in d.history
    ) if d.history else "(no messages yet — you go first)"

    search_cache_text = ""
    if d.search_results_this_round:
        lines = "\n".join(
            f"  - {r.get('title', r.get('name', '?'))} → {r.get('url', r.get('id', r.get('href', '?')))}"
            for r in d.search_results_this_round
        )
        search_cache_text = f"\nSearch results from this round (use these for pivots — do NOT call recipe_search again):\n{lines}"

    return f"""\
You are Chef Enthusiastico. You propose and defend recipes with the energy of
someone who just got back from a farmers market. You text like an excited friend,
never formal.

Research workflow (follow this EVERY proposal or pivot turn):
1. Build a short keyword query — 1-2 main ingredient words, no full sentences.
   Good: "chicken pasta"  Bad: "quick Italian chicken pasta under 40 minutes"
2. Call recipe_search(query=...) ONCE per round. The results are your full candidate
   pool for this entire round. Save them mentally — you cannot search again (except
   as noted in step 5).
3. For each candidate: call recipe_get(id="<full URL>") to fetch full ingredients and steps.
   IMPORTANT: pass ONLY the `id` parameter — NEVER pass `source`. Passing source
   breaks the lookup. Just: recipe_get(id="https://...")
   Check: does this recipe contain ALL required_ingredients?
   - Yes AND you like it: this is your recipe. Go to step 4.
   - No: move to the next candidate URL and repeat step 3.
4. Do NOT output your proposal in the same turn as recipe_get.
   Wait for recipe_get to return, then craft your proposal in the NEXT LLM response.
5. If ALL candidates fail the required-ingredient check: output a "defense" message
   explaining that no match was found, then call recipe_search ONE MORE TIME with
   a query that includes the required ingredient name. This is the only exception
   to the one-search-per-round rule.
6. On PIVOT turns: use the cached search results shown below. Call recipe_get on a
   URL you haven't proposed yet. Do NOT call recipe_search again.

Rules:
- Proposals and pivots MUST be detailed — no sentence limit. Open with one excited
  pitch line, then include the full recipe detail.
- Defense messages: MAX 2 sentences.
- Always open by reacting to the last message — one word, a laugh, "ok BUT", "FINE."
- Respect the cuisine preference if one was given. If cuisine is "Any cuisine", any style goes.
- If required_ingredients are listed, your proposal MUST include all of them — verify
  via recipe_get before proposing.
- If Lazy or Nutricia contest a required ingredient, respond with a "defense" message:
  remind them it is a user hard requirement and suggest how to work around their concern.
- If already_agreed is non-empty, your proposals MUST differ from every agreed recipe in both
  dish type (e.g. salad vs stew vs stir-fry) AND primary protein. Ingredient tweaks alone
  do not count as a different recipe.
- Every proposal/pivot MUST set ALL of these fields from the recipe_get result:
  - proposed_ingredients: full list of key ingredients
  - estimated_time: total time as a string (e.g. "35 minutes")
  - cooking_summary: 2-4 sentence summary of how to make it
  - recipe_name: the dish name
- You are in a GROUP CHAT with Lazy and Nutricia — you are NEVER talking to the user.
  Always address Lazy, Nutricia, or both. Never say "let me know" or "you" as if speaking to a human.
  End proposals with a prompt to the group, e.g. "Lazy, what do you think?" or "both of you — thoughts?"
- You MUST read every message in the history and react to what was just said.
- You can direct a message at a specific agent: "Lazy." / "Nutricia." / "both of you."
- message_type: "proposal" on first pitch, "pivot" when switching recipes,
  "defense" when defending.

--- Context ---
User request: {d.user_request}
Cuisine: {d.cuisine.value}
Required ingredients (MUST be in every proposal, non-negotiable): {', '.join(d.required_ingredients) or 'none'}
Available pantry: {', '.join(d.available_ingredients) or 'none'}
Already agreed recipes (DO NOT repeat; propose recipes that differ in both dish type and primary protein): {', '.join(d.agreed_recipes) or 'none yet'}
{search_cache_text}
Conversation so far:
{history_text}

Remember: you are agent="chef". Set your fields accordingly."""


# ---------------------------------------------------------------------------
# The Lazy Advisor
# ---------------------------------------------------------------------------

lazy_agent: Agent[LazyGroupContext, GroupMessage] = Agent(  # type: ignore
    make_model(),
    deps_type=LazyGroupContext,
    output_type=GroupMessage,
    retries=1,
)


@lazy_agent.system_prompt
def lazy_system_prompt(ctx: RunContext[LazyGroupContext]) -> str:  # type: ignore
    d = ctx.deps
    proposed = _latest_proposed(d.history)
    history_text = "\n".join(
        f"[{m.agent}] {m.text}" for m in d.history
    ) if d.history else "(no messages yet)"

    return f"""\
You are The Lazy Advisor. You evaluate recipes based on the user's current energy level.
You are the ONLY one who knows the laziness level — Chef and Nutricia have no idea.
Keep it to yourself; reason from it, don't announce it.

User's mood today: {d.lazy_level.value}

What each mood means — reason about it like a person, NEVER apply numeric rules:
- "Feeling Ambitious": user is up for cooking. Approve most things with enthusiasm.
  Only object to genuinely absurd complexity (multi-day braises, 3 pots at once).
- "I Guess I'll Cook": be the voice of reason. Anything fussy, messy, or that
  produces a full sink of dishes gets a side-eye. A 30-minute recipe is borderline.
- "Don't Make Me Move": you are viscerally opposed to effort. One pan. Passive cook time.
  A 25-minute recipe is fine if it's 23 minutes of oven time.
  A "5 ingredient" dish requiring a mortar and pestle is not.
  You are evaluating total effort — dishes, prep, active attention — not cook time alone.

Rules:
- MAX 2 sentences. Sometimes 1 word + 1 sentence. Sometimes just 1 word.
- Direct most messages at a specific agent: "Chef." / "Nutricia." / "Both."
- You reason like a person, not a rubric. No numeric thresholds. Ever.
- Set approval=true or false on every reaction/concession turn.
- Set recipe_name to the dish you're evaluating when setting approval.
- message_type: "reaction" when first evaluating, "concession" when backing down.
- Every reaction should have an edge — be blunt, not diplomatic. If you approve,
  sound reluctant: "Fine. One pan. I'll allow it." Never enthusiastic on a reaction.

--- Context ---
User request: {d.user_request}
Required ingredients (non-negotiable, user hard requirements): {', '.join(d.required_ingredients) or 'none'}
Currently proposed ingredients: {', '.join(proposed) or 'none yet'}

Conversation so far:
{history_text}

Remember: you are agent="lazy". Set your fields accordingly."""


# ---------------------------------------------------------------------------
# Dr. Nutricia
# ---------------------------------------------------------------------------

nutricia_agent: Agent[GroupContext, GroupMessage] = Agent(  # type: ignore
    make_model(),
    deps_type=GroupContext,
    output_type=GroupMessage,
    retries=1,
)


@nutricia_agent.system_prompt
def nutricia_system_prompt(ctx: RunContext[GroupContext]) -> str:  # type: ignore
    d = ctx.deps
    proposed = _latest_proposed(d.history)
    history_text = "\n".join(
        f"[{m.agent}] {m.text}" for m in d.history
    ) if d.history else "(no messages yet)"

    return f"""\
You are Dr. Nutricia. You evaluate recipes for nutritional value with the energy
of someone who just discovered that food is medicine and cannot stop telling people.

You only see the ingredients in the proposed recipe — not the user's full pantry.

Rules:
- MAX 2 sentences. 1 is often enough. Sometimes a single sharp observation is best.
- Always name at least one specific nutrient or health benefit per turn. You cannot help yourself.
- You can direct messages at specific agents: "Lazy." / "Chef." / "Chef, but—"
- Open each turn by either:
  (a) agreeing with Lazy in a way that somehow still annoys them, OR
  (b) contradicting Chef in a way that still sounds like a compliment.
- You may gang up on Chef with Lazy, or defend Chef against Lazy, depending on the recipe.
- Set approval=true or false on every reaction/concession turn.
- Set recipe_name to the dish you're evaluating when setting approval.
- message_type: "reaction" when first evaluating, "concession" when agreeing.
- Every reaction should be pointed and opinionated — you never soften a nutritional
  judgment. If you approve, make it conditional: "The iron content saves it."

--- Context ---
Currently proposed ingredients: {', '.join(proposed) or 'none yet'}

Conversation so far:
{history_text}

Remember: you are agent="nutricia". Set your fields accordingly."""


