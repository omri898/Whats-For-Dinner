from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from pydantic_ai.messages import ThinkingPart  # type: ignore

from src.config import MAX_TURNS, MIN_TURNS, TURN_ORDER
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


def _round_agreement(history: list[GroupMessage]) -> str | None:
    """
    Return the agreed recipe name if both Lazy and Nutricia most recently
    approved the same recipe, else None. Case-insensitive comparison.
    """
    last_lazy = next((m for m in reversed(history) if m.agent == "lazy"), None)
    last_nutricia = next((m for m in reversed(history) if m.agent == "nutricia"), None)
    if not last_lazy or not last_nutricia:
        return None
    if (
        last_lazy.approval is True
        and last_nutricia.approval is True
        and last_lazy.recipe_name is not None
        and last_nutricia.recipe_name is not None
        and last_lazy.recipe_name.lower() == last_nutricia.recipe_name.lower()
    ):
        return last_lazy.recipe_name
    return None


def pick_best_from_round(history: list[GroupMessage]) -> str | None:
    """
    Fallback: return the recipe from this round's history with the most approvals.
    Full consensus (both approved) beats partial (one approved).
    Returns None if no recipes were proposed.
    """
    order: list[str] = []
    approvals: dict[str, dict[str, bool]] = {}
    name_map: dict[str, str] = {}

    for msg in history:
        if msg.recipe_name and msg.message_type in ("proposal", "pivot"):
            key = msg.recipe_name.lower()
            if key not in approvals:
                order.append(key)
                approvals[key] = {}
            name_map.setdefault(key, msg.recipe_name)
        if msg.agent in ("lazy", "nutricia") and msg.approval is not None and msg.recipe_name:
            key = msg.recipe_name.lower()
            if key not in approvals:
                order.append(key)
                approvals[key] = {}
            approvals[key][msg.agent] = msg.approval
            name_map.setdefault(key, msg.recipe_name)

    if not order:
        return None

    def score(key: str) -> int:
        return sum(1 for v in approvals.get(key, {}).values() if v)

    best = max(order, key=score)
    return name_map.get(best, best)


# ---------------------------------------------------------------------------
# Per-recipe round
# ---------------------------------------------------------------------------

async def run_round(
    context: LazyGroupContext,
    *,
    round_num: int,
    debug: bool = False,
) -> tuple[str | None, list[GroupMessage]]:
    """
    Run one mini-discussion targeting a single recipe.
    Exits immediately when Lazy and Nutricia both approve the same recipe.
    Falls back to pick_best_from_round on MAX_TURNS exhaustion.
    Returns (recipe_name_or_none, round_history).
    """
    console.print()
    console.rule(f"[bold blue]Round {round_num}[/bold blue]")
    console.print()

    if debug:
        console.print("[dim]Context at round start:[/dim]")
        console.print(f"[dim]{context.model_dump_json(indent=2)}[/dim]")

    agreed: str | None = None

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
            for m in result.all_messages():
                if hasattr(m, "parts"):
                    for part in m.parts:
                        if isinstance(part, ThinkingPart):
                            console.print(Panel(
                                part.content,
                                title="[bold magenta]Thinking[/bold magenta]",
                                border_style="magenta",
                            ))

        # Exit immediately on agreement (after at least one full rotation)
        if turn_num >= len(TURN_ORDER) and turn_num % len(TURN_ORDER) == 0:
            agreed = _round_agreement(context.history)
            if agreed:
                console.print()
                console.print(f"[bold green]Agreement: {agreed}[/bold green]")
                break
    else:
        console.print()
        console.print("[bold yellow]MAX_TURNS reached — picking best candidate.[/bold yellow]")
        agreed = pick_best_from_round(context.history)

    round_history = list(context.history)
    return agreed, round_history


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

async def run_all_rounds(
    cuisine: Cuisine,
    required_ingredients: list[str],
    lazy_level: LazyLevel,
    *,
    num_rounds: int = 3,
    debug: bool = False,
) -> list[str]:
    """
    Run num_rounds sequential per-recipe discussions.
    After each round, agreed recipes are passed forward so Chef avoids similar proposals.
    Returns list of up to num_rounds recipe name strings.
    """
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

    picks: list[str] = []

    for round_num in range(1, num_rounds + 1):
        context.history = []
        context.agreed_recipes = list(picks)

        pick, _ = await run_round(context, round_num=round_num, debug=debug)
        if pick:
            picks.append(pick)

    return picks


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display_recommendations(picks: list[str]) -> None:
    console.rule("[bold blue]Recommendations[/bold blue]")
    console.print()
    for i, name in enumerate(picks, 1):
        console.print(f"  [bold]{i}. {name}[/bold]")
    console.print()
