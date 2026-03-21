from __future__ import annotations

import httpx
from rich import print as rprint
from rich.table import Table

from pydantic_ai import Agent, RunContext  # type: ignore
from pydantic_ai.models.openai import OpenAIChatModel  # type: ignore
from pydantic_ai.providers.openai import OpenAIProvider  # type: ignore

from src.config import MODEL_NAME, VLLM_BASE
from src.models import (
    GroupContext,
    GroupMessage,
    LazyGroupContext,
)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_model() -> OpenAIChatModel:
    return OpenAIChatModel(
        MODEL_NAME,
        provider=OpenAIProvider(
            base_url=VLLM_BASE,
            api_key="dummy",
        ),
    )


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
    retries=1,
)


@chef_agent.system_prompt
def chef_system_prompt(ctx: RunContext[GroupContext]) -> str:  # type: ignore
    d = ctx.deps
    history_text = "\n".join(
        f"[{m.agent}] {m.text}" for m in d.history
    ) if d.history else "(no messages yet — you go first)"

    return f"""\
You are Chef Enthusiastico. You propose and defend recipes with the energy of
someone who just got back from a farmers market. You text like an excited friend,
never formal.

Rules:
- MAX 2 sentences. One sentence is usually better. Sometimes a single word + a dish name is enough.
- Always open by reacting to the last message — one word, a laugh, "ok BUT", "FINE."
- On your first turn, name the dish immediately, then pitch it in one breathless line.
- On pivot turns (after being rejected), admit it fast ("fair.") and immediately name a new dish.
- Respect the cuisine preference if one was given. If cuisine is "I Don't Mind", ignore it.
- If required_ingredients are listed, your proposal MUST include all of them — call this out.
- If Lazy or Nutricia contest a required ingredient, respond with a "defense" message: remind
  them it is a user hard requirement and suggest how to work around their concern
  (e.g. a pairing, or a substitution elsewhere in the dish).
- If already_agreed is non-empty, your proposals MUST differ from every agreed recipe in both
  dish type (e.g. salad vs stew vs stir-fry) AND primary protein. Ingredient tweaks alone
  do not count as a different recipe.
- Always set proposed_ingredients on proposal/pivot turns — list the key ingredients.
- You MUST read every message in the history and react to what was just said.
- You can direct a message at a specific agent: "Lazy." / "Nutricia." / "both of you."
- Your recipe_name field must always be set on proposal/pivot turns.
- message_type: "proposal" on first pitch, "pivot" when switching recipes,
  "defense" when defending, "sting" when being sarcastic.

--- Context ---
User request: {d.user_request}
Cuisine: {d.cuisine.value}
Required ingredients (MUST be in every proposal, non-negotiable): {', '.join(d.required_ingredients) or 'none'}
Available pantry: {', '.join(d.available_ingredients) or 'none'}
Already agreed recipes (DO NOT repeat; propose recipes that differ in both dish type and primary protein): {', '.join(d.agreed_recipes) or 'none yet'}

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
- message_type: "reaction" when first evaluating, "concession" when backing down,
  "sting" when being brutal.

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
- message_type: "reaction" when first evaluating, "concession" when agreeing,
  "sting" when lecturing.

--- Context ---
Currently proposed ingredients: {', '.join(proposed) or 'none yet'}

Conversation so far:
{history_text}

Remember: you are agent="nutricia". Set your fields accordingly."""


