from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from pydantic_ai.messages import ThinkingPart  # type: ignore

from src.config import MAX_TURNS, MIN_TURNS, MIN_PROPOSALS, TURN_ORDER
from src.models import (
    Cuisine,
    FinalRecommendations,
    GroupMessage,
    LazyGroupContext,
    LazyLevel,
    SupervisorContext,
)
from src.agents import chef_agent, lazy_agent, nutricia_agent, supervisor_agent

console = Console()

AGENT_DISPLAY = {
    "chef": ("Chef Enthusiastico", "red"),
    "lazy": ("The Lazy Advisor", "yellow"),
    "nutricia": ("Dr. Nutricia", "green"),
}

AGENT_MAP = {
    "chef": chef_agent,
    "lazy": lazy_agent,
    "nutricia": nutricia_agent,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_ingredients() -> list[str]:
    path = Path(__file__).parent.parent / "data" / "ingredients.json"
    with open(path) as f:
        return json.load(f)["ingredients"]


def agreement_reached(history: list[GroupMessage], min_proposals: int) -> bool:
    """
    True when:
    - Chef has proposed at least min_proposals different recipes
    - Both lazy and nutricia approved the SAME recipe name in their most recent turns
    Recipe name comparison is case-insensitive.
    """
    proposals = [m for m in history if m.recipe_name and m.message_type in ("proposal", "pivot")]
    unique_recipes = {m.recipe_name.lower() for m in proposals}
    if len(unique_recipes) < min_proposals:
        return False
    last_lazy = next((m for m in reversed(history) if m.agent == "lazy"), None)
    last_nutricia = next((m for m in reversed(history) if m.agent == "nutricia"), None)
    if not last_lazy or not last_nutricia:
        return False
    return (
        last_lazy.approval is True
        and last_nutricia.approval is True
        and last_lazy.recipe_name is not None
        and last_nutricia.recipe_name is not None
        and last_lazy.recipe_name.lower() == last_nutricia.recipe_name.lower()
    )


def _print_message(msg: GroupMessage, turn: int, debug: bool = False) -> None:
    """Print a single discussion message with Rich formatting."""
    if msg.agent == "system":
        console.print(f"[dim]{msg.text}[/dim]")
        return

    name, color = AGENT_DISPLAY.get(msg.agent, (msg.agent, "white"))
    subtitle = (
        f"[dim]Turn {turn} · {msg.message_type} · → {msg.directed_at}"
        f"{' · ' + msg.recipe_name if msg.recipe_name else ''}[/dim]"
    )
    console.print(Panel(
        msg.text,
        title=f"[bold {color}]{name}[/bold {color}]",
        subtitle=subtitle,
        border_style=color,
    ))

    if debug and msg.proposed_ingredients:
        console.print(f"  [dim]proposed_ingredients: {msg.proposed_ingredients}[/dim]")


# ---------------------------------------------------------------------------
# Discussion loop
# ---------------------------------------------------------------------------

async def run_discussion(
    cuisine: Cuisine,
    required_ingredients: list[str],
    lazy_level: LazyLevel,
    *,
    debug: bool = False,
) -> list[GroupMessage]:
    """Run the multi-agent discussion. Returns the full message history."""
    available = load_ingredients()

    if required_ingredients:
        user_request = (
            f"The user wants to prepare dinner of cuisine {cuisine.value}, "
            f"using {', '.join(required_ingredients)}."
        )
    else:
        user_request = f"The user wants to prepare dinner of cuisine {cuisine.value}."

    context = LazyGroupContext(
        user_request=user_request,
        cuisine=cuisine,
        required_ingredients=list(required_ingredients),
        available_ingredients=available,
        locked_ingredients=[],
        lazy_level=lazy_level,
    )

    if debug:
        console.print("[dim]Initial context:[/dim]")
        console.print(f"[dim]{context.model_dump_json(indent=2)}[/dim]")

    console.print()
    console.rule("[bold blue]Discussion[/bold blue]")
    console.print()

    for turn_num in range(1, MAX_TURNS + 1):
        agent_name = TURN_ORDER[(turn_num - 1) % len(TURN_ORDER)]
        agent = AGENT_MAP[agent_name]
        display_name = AGENT_DISPLAY[agent_name][0]

        with console.status(f"[dim]{display_name} is thinking...[/dim]"):
            result = await agent.run("Your turn.", deps=context)

        msg = result.output
        context.history.append(msg)
        _print_message(msg, turn_num, debug=debug)

        if debug:
            # Show thinking tokens
            for m in result.all_messages():
                if hasattr(m, "parts"):
                    for part in m.parts:
                        if isinstance(part, ThinkingPart):
                            console.print(Panel(
                                part.content,
                                title="[bold magenta]Thinking[/bold magenta]",
                                border_style="magenta",
                            ))

        # Lock mechanic: Chef locks a required ingredient
        if agent_name == "chef" and msg.message_type == "lock":
            # Sync locked_ingredients with required_ingredients
            for ing in context.required_ingredients:
                if ing not in context.locked_ingredients:
                    context.locked_ingredients.append(ing)
            # Inject system note
            system_note = GroupMessage(
                agent="system",
                message_type="system",
                text=f"[System] Locked ingredients: {', '.join(context.locked_ingredients)}. "
                     "These are user hard requirements — do not contest them.",
                directed_at="all",
            )
            context.history.append(system_note)
            _print_message(system_note, turn_num, debug=debug)

        # Check agreement after each full rotation past MIN_TURNS
        if turn_num >= MIN_TURNS and turn_num % len(TURN_ORDER) == 0:
            if agreement_reached(context.history, MIN_PROPOSALS):
                console.print()
                console.print("[bold green]Agreement reached![/bold green]")
                break
    else:
        console.print()
        console.print("[bold yellow]MAX_TURNS reached — no full agreement.[/bold yellow]")

    console.print()
    return context.history


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

async def run_supervisor(history: list[GroupMessage]) -> FinalRecommendations:
    """Run the Supervisor agent on the discussion history."""
    ctx = SupervisorContext(history=history)

    console.rule("[bold blue]Supervisor[/bold blue]")
    console.print()

    with console.status("[dim]Supervisor is reviewing the discussion...[/dim]"):
        result = await supervisor_agent.run("Summarize.", deps=ctx)

    recs = result.output

    # Print recommendations
    console.print(Panel(
        recs.supervisor_verdict,
        title="[bold blue]Supervisor's Verdict[/bold blue]",
        border_style="blue",
    ))

    for i, pick in enumerate(recs.picks, 1):
        console.print(f"  [bold]{i}. {pick.recipe_name}[/bold] — {pick.why_it_won}")

    console.print(f"\n  [italic]{recs.vibe_check}[/italic]\n")

    return recs
