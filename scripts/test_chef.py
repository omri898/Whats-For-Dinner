"""Phase 2 test: verify vLLM health check and Chef agent alone."""

import asyncio
import json
from pathlib import Path

from rich import print as rprint
from rich.panel import Panel

from src.agents import check_vllm, chef_agent
from src.models import GroupContext, Cuisine


async def main() -> None:
    await check_vllm()

    # Load pantry
    pantry_path = Path(__file__).parent.parent / "data" / "ingredients.json"
    with open(pantry_path) as f:
        available = json.load(f)["ingredients"]

    context = GroupContext(
        user_request="The user wants to prepare dinner of cuisine Italian, using eggs, feta cheese.",
        cuisine=Cuisine.ITALIAN,
        required_ingredients=["eggs", "feta cheese"],
        available_ingredients=available,
    )

    rprint("\n[bold]Running Chef Enthusiastico...[/bold]\n")
    async with chef_agent:
        result = await chef_agent.run("Your turn.", deps=context)
    msg = result.output

    rprint(Panel(
        msg.text,
        title="[bold red]Chef Enthusiastico[/bold red]",
        subtitle=f"[dim]{msg.message_type} · → {msg.directed_at}"
                 f"{' · ' + msg.recipe_name if msg.recipe_name else ''}[/dim]",
        border_style="red",
    ))

    rprint("\n[dim]Raw GroupMessage:[/dim]")
    rprint(msg.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
