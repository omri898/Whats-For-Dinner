"""Phase 2 test: verify vLLM health check and Chef agent alone."""

import asyncio
import json
from pathlib import Path

from rich import print as rprint
from rich.panel import Panel
from rich.console import Console
from pydantic_ai.messages import (  # type: ignore
    ThinkingPart, TextPart, SystemPromptPart, UserPromptPart,
    ToolCallPart, ToolReturnPart,
)

from src.agents import check_vllm, chef_agent
from src.models import GroupContext, Cuisine

console = Console()


def _print_part(part: object) -> None:
    if isinstance(part, SystemPromptPart):
        console.print(Panel(part.content, title="[bold cyan]System Prompt[/bold cyan]", border_style="cyan"))  # type: ignore[union-attr]
    elif isinstance(part, UserPromptPart):
        content = part.content if isinstance(part.content, str) else json.dumps(part.content, indent=2)  # type: ignore[union-attr]
        console.print(Panel(content, title="[bold blue]User Prompt[/bold blue]", border_style="blue"))
    elif isinstance(part, ThinkingPart):
        console.print(Panel(part.content, title="[bold magenta]Thinking[/bold magenta]", border_style="magenta"))  # type: ignore[union-attr]
    elif isinstance(part, ToolCallPart):
        args_str = part.args if isinstance(part.args, str) else json.dumps(part.args, indent=2)  # type: ignore[union-attr]
        console.print(Panel(
            f"tool: {part.tool_name}\nargs: {args_str}",  # type: ignore[union-attr]
            title="[bold yellow]Tool Call[/bold yellow]",
            border_style="yellow",
        ))
    elif isinstance(part, ToolReturnPart):
        console.print(Panel(
            str(part.content),  # type: ignore[union-attr]
            title=f"[bold yellow]Tool Return · {part.tool_name}[/bold yellow]",  # type: ignore[union-attr]
            border_style="yellow",
        ))
    elif isinstance(part, TextPart) and part.content.strip():  # type: ignore[union-attr]
        console.print(Panel(part.content, title="[bold white]Raw LLM Text[/bold white]", border_style="white"))  # type: ignore[union-attr]


async def main() -> None:
    await check_vllm()

    # Load pantry
    pantry_path = Path(__file__).parent.parent / "data" / "ingredients.json"
    with open(pantry_path) as f:
        available = json.load(f)["ingredients"]

    context = GroupContext(
        user_request="The user wants to prepare dinner of cuisine Mediterranean, using eggs, feta cheese.",
        cuisine=Cuisine.MEDITERRANEAN,
        required_ingredients=["eggs", "feta cheese"],
        available_ingredients=available,
    )

    rprint("\n[bold]Running Chef Enthusiastico (streaming debug)...[/bold]\n")

    seen = 0
    crashed = False
    async with chef_agent:
        async with chef_agent.iter("Your turn.", deps=context) as agent_run:
            try:
                async for _node in agent_run:
                    all_msgs = agent_run.all_messages()
                    for m in all_msgs[seen:]:
                        if hasattr(m, "parts"):
                            for part in m.parts:
                                _print_part(part)
                    seen = len(all_msgs)
            except Exception as exc:
                rprint(f"\n[bold red]Agent crashed: {exc}[/bold red]")
                for m in agent_run.all_messages()[seen:]:
                    if hasattr(m, "parts"):
                        for part in m.parts:
                            _print_part(part)
                crashed = True

    if crashed:
        rprint("\n[yellow]Run did not complete — see crash above.[/yellow]")
        return

    result = agent_run.result
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
