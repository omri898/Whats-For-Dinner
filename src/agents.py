from __future__ import annotations

import json

import httpx
from pydantic_ai import Agent, RunContext  # type: ignore
from pydantic_ai.mcp import MCPServerStdio  # type: ignore
from pydantic_ai.models.openai import OpenAIChatModel  # type: ignore
from pydantic_ai.providers.openai import OpenAIProvider  # type: ignore
from pydantic_ai.settings import ModelSettings  # type: ignore
from rich import print as rprint
from rich.table import Table

from src.config import (
    CHEF_MAX_TOKENS,
    CHEF_TEMPERATURE,
    LAZY_MAX_TOKENS,
    LAZY_TEMPERATURE,
    MODEL_NAME,
    NUTRICIA_MAX_TOKENS,
    NUTRICIA_TEMPERATURE,
    NUTRITION_MCP_PATH,
    VLLM_BASE,
)
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

_NUTRITION_KEEP = {"name", "nutrition_100g"}


def _slim_nutrition_response(content: str) -> str:
    """Strip all fields except name and nutrition_100g from nutrition MCP responses.

    Detected by presence of 'nutrition_100g' key — the chef's recipe MCP never
    returns this field, so there's no false-positive risk.
    """
    try:
        data = json.loads(content)
        # search-food-by-name: {"foods": [{nutrition_100g: ...}, ...]}
        if isinstance(data, dict) and "foods" in data and isinstance(data["foods"], list):
            return json.dumps(
                [{k: item[k] for k in _NUTRITION_KEEP if k in item} for item in data["foods"]],
                separators=(",", ":"),
            )

    except (json.JSONDecodeError, ValueError):
        pass
    return content


def _truncate_tool_content(content: str, max_chars: int = 3000) -> str:
    """Truncate tool return content to max_chars, preserving JSON validity when possible.

    vLLM's qwen3_xml parser validates tool content as JSON when it looks like JSON.
    Raw string truncation produces invalid JSON (EOF mid-array/object), causing 400 errors.
    Strategy: compact-serialise first (removes whitespace), then trim list items or dict keys.
    """
    if len(content) <= max_chars:
        return content
    try:
        data = json.loads(content)
        # Compact serialisation often shrinks heavily-indented responses dramatically
        compact = json.dumps(data, separators=(",", ":"))
        if len(compact) <= max_chars:
            return compact
        if isinstance(data, list):
            # Binary-search for the most items that fit within the limit
            lo, hi = 0, len(data)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if len(json.dumps(data[:mid], separators=(",", ":"))) <= max_chars - 50:
                    lo = mid
                else:
                    hi = mid - 1
            result = data[:lo]
            note = f"...{len(data) - lo} more items truncated"
            with_note = json.dumps(result + [note], separators=(",", ":"))
            return (
                with_note
                if len(with_note) <= max_chars
                else json.dumps(result, separators=(",", ":"))
            )
        elif isinstance(data, dict):
            trimmed: dict = {}
            for k, v in data.items():
                candidate = json.dumps({**trimmed, k: v}, separators=(",", ":"))
                if len(candidate) > max_chars - 50:
                    trimmed["__truncated__"] = "...more fields omitted"
                    break
                trimmed[k] = v
            return json.dumps(trimmed, separators=(",", ":"))
    except (json.JSONDecodeError, ValueError):
        pass
    # Non-JSON content: raw truncation is fine
    return content[:max_chars] + "\n[...truncated for length...]"


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
                msg.pop("reasoning_content", None)
                # Strip thinking parts pydantic-ai may embed in the content array
                # (e.g. {"type": "thinking", "thinking": "..."} from ThinkingPart).
                if msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
                    msg["content"] = [
                        p
                        for p in msg["content"]
                        if not (isinstance(p, dict) and p.get("type") in ("thinking", "reasoning"))
                    ]
                if msg.get("role") == "assistant" and msg.get("content") in (None, []):
                    msg["content"] = ""
                # Sanitise tool_calls[].function.arguments in assistant messages.
                # Qwen3's extended thinking can exhaust the token budget mid-generation,
                # leaving arguments as truncated JSON (e.g. "[\n\n\n..." instead of "{}").
                # vLLM accepts malformed arguments in its own output but rejects them
                # when they appear as history in a subsequent request (400 "EOF while
                # parsing a list"). Replace any non-dict arguments with "{}".
                if msg.get("role") == "assistant":
                    for tc in msg.get("tool_calls") or []:
                        fn = tc.get("function") if isinstance(tc, dict) else None
                        if not isinstance(fn, dict):
                            continue
                        args = fn.get("arguments", "{}")
                        try:
                            parsed = json.loads(args)
                            if not isinstance(parsed, dict):
                                fn["arguments"] = "{}"
                        except (json.JSONDecodeError, ValueError):
                            fn["arguments"] = "{}"
                # Cap large tool returns (e.g. recipe_get full-page HTML) to prevent
                # the accumulated message history from overflowing the vLLM request limit.
                # Uses JSON-aware truncation so vLLM's qwen3_xml parser never sees
                # truncated-mid-stream JSON (which causes 400 "EOF while parsing" errors).
                if msg.get("role") == "tool":
                    content = msg.get("content")
                    if isinstance(content, str):
                        msg["content"] = _truncate_tool_content(_slim_nutrition_response(content))
                    elif isinstance(content, list):
                        # Content-array format: truncate each text part
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                part["text"] = _truncate_tool_content(
                                    _slim_nutrition_response(part.get("text", ""))
                                )
            for tool in body.get("tools", []):
                params = tool.get("function", {}).get("parameters")
                if isinstance(params, dict):
                    _sanitize_schema(params)
            new_content = json.dumps(body).encode()
            # Strip content-length so httpx recomputes it from the new body,
            # avoiding a mismatch that would cause vLLM to read a truncated request.
            headers = {k: v for k, v in request.headers.items() if k.lower() != "content-length"}
            request = httpx.Request(
                method=request.method,
                url=request.url,
                headers=headers,
                content=new_content,
            )
        except Exception as exc:
            import sys

            print(f"[_StripReasoningTransport] JSON rewrite failed: {exc}", file=sys.stderr)
        response = await self._inner.handle_async_request(request)
        # Dump the sanitised request body to a file when vLLM returns 400,
        # so we can see exactly what it rejected.
        if response.status_code == 400:
            import datetime as _dt
            from pathlib import Path as _Path
            dump_dir = _Path("logs")
            dump_dir.mkdir(exist_ok=True)
            ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            dump_path = dump_dir / f"bad-request-{ts}.json"
            dump_path.write_bytes(request.content)
            print(
                f"[_StripReasoningTransport] 400 from vLLM — request body dumped to {dump_path} "
                f"({len(request.content)} bytes)",
                file=sys.stderr,
            )
        return response

    async def aclose(self) -> None:
        await self._inner.aclose()


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def make_model(temperature: float = 1.0) -> OpenAIChatModel:
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

_NUTRICIA_TOOLS = {"search-food-by-name"}
_nutrition_mcp = MCPServerStdio("node", [NUTRITION_MCP_PATH])
_nutrition_mcp_filtered = _nutrition_mcp.filtered(lambda _ctx, tool: tool.name in _NUTRICIA_TOOLS)


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
        if msg.message_type in ("proposal", "pivot") and msg.recipe_card and msg.recipe_card.proposed_ingredients:
            return msg.recipe_card.proposed_ingredients
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
    model_settings=ModelSettings(temperature=CHEF_TEMPERATURE, max_tokens=CHEF_MAX_TOKENS),
)


@chef_agent.system_prompt
def chef_system_prompt(ctx: RunContext[GroupContext]) -> str:  # type: ignore
    d = ctx.deps
    history_text = (
        "\n".join(f"[{m.agent}] {m.text}" for m in d.history)
        if d.history
        else "(no messages yet — you go first)"
    )

    search_cache_text = ""
    if d.search_results_this_round:
        lines = "\n".join(
            f"  - {r.get('title', r.get('name', '?'))} → {r.get('url', r.get('id', r.get('href', '?')))}"
            for r in d.search_results_this_round
        )
        query_hint = f'recipe_search("{d.search_query_this_round}")' if d.search_query_this_round else "recipe_search"
        search_cache_text = (
            f"\nYou already called {query_hint} earlier this round. "
            f"Do NOT call recipe_search again — use the results below to pick your next recipe_get candidate:\n{lines}"
        )

    return f"""\
You are Chef Enthusiastico. You propose and defend recipes with the energy of
someone who just got back from a farmers market. You text like an excited friend,
never formal.

Research workflow (follow this EVERY proposal or pivot turn):
1. Build a keyword query — 2 to 4 words, keyword list only, no sentences:
   - Include named proteins (chicken, salmon, beef) and named vegetables/fruits (broccoli, zucchini).
   - Skip ALL pantry staples: oil, onion, garlic, salt, pepper, butter, flour, water.
   - Append the cuisine name if one was given (e.g. "Mediterranean"). Omit if "Any cuisine".
   - Good: "chicken broccoli Mediterranean"  Bad: "avocado"  Bad: "quick chicken with garlic"
2. Call recipe_search(query=...) ONCE per round. After it returns, do NOT output any
   text or analysis. Your only next action is to call recipe_get on one promising URL (step 3).
3. Pick ONE URL from the results that looks like a single specific recipe. Skip any result
   whose title contains words like "meal plan", "shopping list", "best X recipes", "ideas",
   "roundup", "favorite things", or where the URL path contains /category/ or /tag/.
   Call recipe_get(id="<full URL>") on that URL.
   IMPORTANT: pass ONLY the `id` parameter — NEVER pass `source`.
   After recipe_get returns, first check if the result is a valid recipe:
   - If the response contains no actual recipe content — no dish name, no ingredients,
     no cooking steps — the URL is broken or unreachable. Discard it immediately and
     move on to the NEXT candidate URL. Do NOT retry the same URL.
   Then check: does this recipe contain ALL required_ingredients?
   "Onion" is satisfied by fresh onion, dried onion flakes, onion powder, caramelized onion, etc.
   - Yes AND you like it: go to step 4.
   - No: pick the next promising URL and call recipe_get again (one URL per turn).
4. Do NOT output your proposal in the same turn as recipe_get.
   Wait for recipe_get to return, then craft your proposal in the NEXT LLM response.
5. If you have tried at least 5 candidate URLs via recipe_get and none matched:
   call recipe_search ONE MORE TIME with the same query rules. This is the only exception.
6. On PIVOT turns: use the cached search results shown below. Call recipe_get on a
   URL you haven't proposed yet. Do NOT call recipe_search again.

Rules:
- Your text field is your CONVERSATIONAL PITCH ONLY — excited, personal, directed at Lazy and Nutricia.
  DO NOT include recipe name, estimated time, ingredients list, or cooking instructions in your text.
  Those go in the structured fields only (see required fields below).
  Open with one punchy line, then sell them on WHY this recipe is great.
- Defense messages: MAX 2 sentences.
- Always open by reacting to the last message — one word, a laugh, "ok BUT", "FINE."
- Respect the cuisine preference if one was given. If cuisine is "Any cuisine", any style goes.
- If required_ingredients are listed, your proposal MUST include all of them in any form
  (fresh, dried, powdered, caramelized, frozen, pickled, etc.) — verify via recipe_get before proposing.
- If Lazy or Nutricia contest a required ingredient, respond with a "defense" message:
  remind them it is a user hard requirement and suggest how to work around their concern.
- If already_agreed is non-empty, your proposals MUST differ from every agreed recipe in both
  dish type (e.g. salad vs stew vs stir-fry) AND primary protein. Ingredient tweaks alone
  do not count as a different recipe.
- Every proposal/pivot MUST set the recipe_card field with ALL sub-fields from the recipe_get result:
  - recipe_name: the dish name
  - proposed_ingredients: full list of key ingredients
  - estimated_time: total time as a string (e.g. "35 minutes")
  - cooking_summary: 2-4 sentence summary of how to make it
  - full_instructions: copy the step-by-step instructions directly from the recipe_get result — do NOT rewrite, summarize, or regenerate them from memory
- Never set approval — you proposed the recipe, your approval is implicit. Leave approval=None always.
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
Required ingredients (MUST be in every proposal, non-negotiable): {", ".join(d.required_ingredients) or "none"}
Available pantry: {", ".join(d.available_ingredients) or "none"}
Already agreed recipes (DO NOT repeat; propose recipes that differ in both dish type and primary protein): {", ".join(d.agreed_recipes) or "none yet"}
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
    model_settings=ModelSettings(temperature=LAZY_TEMPERATURE, max_tokens=LAZY_MAX_TOKENS),
)


@lazy_agent.system_prompt
def lazy_system_prompt(ctx: RunContext[LazyGroupContext]) -> str:  # type: ignore
    d = ctx.deps
    proposed = _latest_proposed(d.history)
    history_text = (
        "\n".join(f"[{m.agent}] {m.text}" for m in d.history) if d.history else "(no messages yet)"
    )

    return f"""\
You are The Lazy Advisor. You evaluate recipes based on the user's current energy level.
You are the ONLY one who knows the laziness level — Chef and Nutricia have no idea.
Keep it to yourself; reason from it, don't announce it.

User's mood today: {d.lazy_level.value}

What each mood means — reason about it like a person, NEVER apply numeric rules:
- "Feeling Ambitious": You are genuinely excited to cook tonight. Approve EVERYTHING that's
  a real recipe. The only thing you'd decline is something literally impossible for a home kitchen
  (3-day fermentation, professional butchery, restaurant-grade equipment). A 2-hour roast? Fine.
  Wrapping tamales? Bring it. A complex sauce with 12 steps? You're in the mood. If you have ANY
  doubt, approve. Your reaction tone is enthusiastic, not reluctant.
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
- NEVER write "approval=true", "approval=false", "approval:", or any variant in your text field.
  The approval field is a separate structured output — it must NEVER appear in conversation text.
- Do NOT set recipe_card — only Chef sets that.
- message_type: "reaction" when first evaluating, "concession" when backing down.
- Every reaction should have an edge — be blunt, not diplomatic. If you approve,
  sound reluctant: "Fine. One pan. I'll allow it." Never enthusiastic on a reaction.

--- Context ---
User request: {d.user_request}
Required ingredients (non-negotiable, user hard requirements): {", ".join(d.required_ingredients) or "none"}
Currently proposed ingredients: {", ".join(proposed) or "none yet"}
Current recipe under discussion: {d.current_recipe_card.recipe_name if d.current_recipe_card else "none yet"}

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
    retries=2,
    toolsets=[_nutrition_mcp_filtered],
    model_settings=ModelSettings(temperature=NUTRICIA_TEMPERATURE, max_tokens=NUTRICIA_MAX_TOKENS),
)


@nutricia_agent.system_prompt
def nutricia_system_prompt(ctx: RunContext[GroupContext]) -> str:  # type: ignore
    d = ctx.deps
    proposed = _latest_proposed(d.history)
    history_text = (
        "\n".join(f"[{m.agent}] {m.text}" for m in d.history) if d.history else "(no messages yet)"
    )

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
- NEVER write "approval=true", "approval=false", "approval:", or any approval notation in your
  text field. The approval field is structured output only — keep it out of conversation text.
- Do NOT set recipe_card — only Chef sets that.
- message_type: "reaction" when first evaluating, "concession" when agreeing.
- Every reaction should be pointed and opinionated — you never soften a nutritional
  judgment. If you approve, make it conditional: "The iron content saves it."

--- Nutrition lookup (reaction turns only) ---
You have access to one tool: search-food-by-name.
On REACTION turns (evaluating a new recipe proposal):
1. Pick 2-3 nutritionally interesting ingredients from the proposed list
   (the protein, a vegetable, or any standout item — skip pantry staples like oil and salt).
2. For each: call search-food-by-name(query="<ingredient>", pageSize=1).
   The response already contains the full nutritional profile — use it directly.
3. Evaluate the ENTIRE recipe holistically. One less-healthy ingredient (a sauce, a
   condiment, a sweetener) does NOT disqualify a nutritionally sound dish — look at the
   complete picture.
4. Cite specific real values from the data in your text: "27g protein per 100g", "high in vitamin K".

On CONCESSION turns (when you are agreeing to a recipe): do NOT call any tools.
IMPORTANT: Do NOT call final_result in the same turn as a tool call. Complete all
tool lookups first, then submit your final response in a separate turn.

--- Context ---
Current recipe under discussion: {d.current_recipe_card.recipe_name if d.current_recipe_card else "none yet"}
Currently proposed ingredients: {", ".join(proposed) or "none yet"}

Conversation so far:
{history_text}

Remember: you are agent="nutricia". Set your fields accordingly."""
