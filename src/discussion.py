from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from pydantic_ai.messages import ThinkingPart  # type: ignore

from src.config import MAX_TURNS, MIN_TURNS, MIN_PROPOSALS, TURN_ORDER
from src.models import (
    Cuisine,
    GroupMessage,
    LazyGroupContext,
    LazyLevel,
)
from src.agents import chef_agent, lazy_agent, nutricia_agent

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
# Recommendations
# ---------------------------------------------------------------------------

def pick_recipes(history: list[GroupMessage], n: int = 3) -> list[str]:
    """
    Deterministically pick up to n recipe recommendations from the discussion history.

    For each recipe name seen in the discussion, track the most recent approval
    from Lazy and Nutricia. Full-consensus picks (both approved) are ranked first,
    then partial picks (one approved), ordered by first appearance in the discussion.
    Returns exactly n names, padding with partials if fewer than n reached consensus.
    """
    # Ordered list of recipe names by first appearance
    order: list[str] = []
    # recipe_name (lower) -> most recent approval per reactor
    approvals: dict[str, dict[str, bool]] = {}

    for msg in history:
        if msg.recipe_name and msg.message_type in ("proposal", "pivot"):
            key = msg.recipe_name.lower()
            if key not in approvals:
                order.append(key)
                approvals[key] = {}
        if msg.agent in ("lazy", "nutricia") and msg.approval is not None and msg.recipe_name:
            key = msg.recipe_name.lower()
            if key not in approvals:
                order.append(key)
                approvals[key] = {}
            approvals[key][msg.agent] = msg.approval

    def score(key: str) -> int:
        a = approvals.get(key, {})
        return sum(1 for v in a.values() if v)

    full = [k for k in order if score(k) == 2]
    partial = [k for k in order if score(k) == 1]
    candidates = full + partial

    # Recover original casing from history
    name_map: dict[str, str] = {}
    for msg in history:
        if msg.recipe_name:
            name_map.setdefault(msg.recipe_name.lower(), msg.recipe_name)

    picks = [name_map.get(k, k) for k in candidates[:n]]
    # Pad with any remaining recipe names if we somehow still have fewer than n
    remaining = [name_map.get(k, k) for k in order if k not in candidates[:n]]
    picks += remaining[:max(0, n - len(picks))]
    return picks[:n]


def display_recommendations(picks: list[str]) -> None:
    console.rule("[bold blue]Recommendations[/bold blue]")
    console.print()
    for i, name in enumerate(picks, 1):
        console.print(f"  [bold]{i}. {name}[/bold]")
    console.print()
