"""Phase 3 test: all three agents with fake history, one round each."""

import asyncio
import json
from pathlib import Path

from rich import print as rprint
from rich.panel import Panel

from src.agents import check_vllm, chef_agent, lazy_agent, nutricia_agent
from src.models import GroupMessage, LazyGroupContext, Cuisine, LazyLevel


AGENT_STYLES = {
    "chef": ("Chef Enthusiastico", "red"),
    "lazy": ("The Lazy Advisor", "yellow"),
    "nutricia": ("Dr. Nutricia", "green"),
}


def print_msg(msg: GroupMessage, turn: int) -> None:
    name, color = AGENT_STYLES.get(msg.agent, (msg.agent, "white"))
    subtitle = (
        f"[dim]Turn {turn} · {msg.message_type} · → {msg.directed_at}"
        f"{' · ' + msg.recipe_name if msg.recipe_name else ''}[/dim]"
    )
    rprint(Panel(msg.text, title=f"[bold {color}]{name}[/bold {color}]",
                 subtitle=subtitle, border_style=color))


async def main() -> None:
    await check_vllm()

    pantry_path = Path(__file__).parent.parent / "data" / "ingredients.json"
    with open(pantry_path) as f:
        available = json.load(f)["ingredients"]

    # Fake history: Chef proposed, now Lazy and Nutricia react
    fake_history = [
        GroupMessage(
            agent="chef", message_type="proposal",
            text="Shakshuka! Eggs poached in spiced tomato — one pan, done in 20 minutes.",
            recipe_name="Shakshuka",
            proposed_ingredients=["eggs", "canned tomatoes", "garlic", "olive oil", "feta cheese"],
            directed_at="all",
        ),
    ]

    context = LazyGroupContext(
        user_request="The user wants to prepare dinner of cuisine Italian, using eggs, feta cheese.",
        cuisine=Cuisine.ITALIAN,
        required_ingredients=["eggs", "feta cheese"],
        available_ingredients=available,
        lazy_level=LazyLevel.COUCH,
        history=list(fake_history),
    )

    # Print fake history
    rprint("\n[bold]--- Fake history ---[/bold]")
    print_msg(fake_history[0], 1)

    # Run Lazy
    rprint("\n[bold]Running Lazy...[/bold]")
    lazy_result = await lazy_agent.run("Your turn.", deps=context)
    lazy_msg = lazy_result.output
    context.history.append(lazy_msg)
    print_msg(lazy_msg, 2)

    # Run Nutricia
    rprint("\n[bold]Running Nutricia...[/bold]")
    nutricia_result = await nutricia_agent.run("Your turn.", deps=context)
    nutricia_msg = nutricia_result.output
    context.history.append(nutricia_msg)
    print_msg(nutricia_msg, 3)

    # Run Chef again
    rprint("\n[bold]Running Chef (round 2)...[/bold]")
    chef_result = await chef_agent.run("Your turn.", deps=context)
    chef_msg = chef_result.output
    context.history.append(chef_msg)
    print_msg(chef_msg, 4)

    # Dump all messages
    rprint("\n[bold]--- All messages ---[/bold]")
    for i, m in enumerate(context.history, 1):
        rprint(f"[dim]{i}. [{m.agent}] {m.message_type}: {m.text[:80]}...[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
