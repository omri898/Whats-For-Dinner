from __future__ import annotations

import argparse
import asyncio
import sys

from rich.console import Console
from rich.columns import Columns
from rich.text import Text

from src.agents import check_vllm
from src.discussion import load_ingredients, run_discussion, pick_recipes, display_recommendations
from src.models import Cuisine, LazyLevel

console = Console()


# ---------------------------------------------------------------------------
# Interactive prompts (no free-text input)
# ---------------------------------------------------------------------------

def prompt_lazy_level() -> LazyLevel:
    levels = list(LazyLevel)
    console.print("\n[bold]How lazy are you tonight?[/bold]\n")
    for i, level in enumerate(levels, 1):
        default = "  (default)" if level == LazyLevel.MEDIUM else ""
        console.print(f"  {i}. {level.value}{default}")
    console.print()

    choice = input("Choose [1-3, default=2]: ").strip()
    if choice in ("1", "2", "3"):
        return levels[int(choice) - 1]
    return LazyLevel.MEDIUM


def prompt_cuisine() -> Cuisine:
    cuisines = list(Cuisine)
    console.print("\n[bold]What cuisine?[/bold]\n")

    # 3-column layout
    col_size = 10
    lines: list[str] = []
    for i, c in enumerate(cuisines, 1):
        default = " (default)" if c == Cuisine.I_DONT_MIND else ""
        lines.append(f"{i:2}. {c.value}{default}")

    # Split into 3 columns
    n = len(lines)
    rows = (n + 2) // 3
    col1 = lines[:rows]
    col2 = lines[rows:2 * rows]
    col3 = lines[2 * rows:]

    for i in range(rows):
        parts = []
        if i < len(col1):
            parts.append(f"{col1[i]:<28}")
        if i < len(col2):
            parts.append(f"{col2[i]:<28}")
        if i < len(col3):
            parts.append(col3[i])
        console.print("  " + "".join(parts))

    console.print()
    choice = input(f"Choose [1-{len(cuisines)}, default={len(cuisines)}]: ").strip()
    try:
        idx = int(choice)
        if 1 <= idx <= len(cuisines):
            return cuisines[idx - 1]
    except ValueError:
        pass
    return Cuisine.I_DONT_MIND


def prompt_ingredients() -> list[str]:
    ingredients = load_ingredients()
    console.print("\n[bold]What must be in the recipe?[/bold]\n")

    # 3-column layout
    lines = [f"{i:2}. {ing}" for i, ing in enumerate(ingredients, 1)]
    n = len(lines)
    rows = (n + 2) // 3
    col1 = lines[:rows]
    col2 = lines[rows:2 * rows]
    col3 = lines[2 * rows:]

    for i in range(rows):
        parts = []
        if i < len(col1):
            parts.append(f"{col1[i]:<24}")
        if i < len(col2):
            parts.append(f"{col2[i]:<24}")
        if i < len(col3):
            parts.append(col3[i])
        console.print("  " + "".join(parts))

    console.print()
    choice = input("Enter numbers (comma-separated) or press Enter to skip: ").strip()
    if not choice:
        return []

    selected: list[str] = []
    for part in choice.split(","):
        part = part.strip()
        try:
            idx = int(part)
            if 1 <= idx <= len(ingredients):
                selected.append(ingredients[idx - 1])
        except ValueError:
            pass
    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def async_main(debug: bool = False) -> None:
    # Health check first
    try:
        await check_vllm()
    except ConnectionError as e:
        console.print(f"[bold red]{e}[/bold red]")
        sys.exit(1)

    # Interactive prompts
    lazy_level = prompt_lazy_level()
    cuisine = prompt_cuisine()
    required = prompt_ingredients()

    console.print(f"\n[bold]Laziness:[/bold] {lazy_level.value}")
    console.print(f"[bold]Cuisine:[/bold] {cuisine.value}")
    console.print(f"[bold]Required:[/bold] {', '.join(required) or 'none'}")

    # Run discussion
    history = await run_discussion(
        cuisine=cuisine,
        required_ingredients=required,
        lazy_level=lazy_level,
        debug=debug,
    )

    # Pick and display recommendations
    picks = pick_recipes(history)
    display_recommendations(picks)

    if debug:
        console.print("\n[dim]--- Debug: raw context JSON would appear here ---[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(description="What's for dinner? Multi-agent dinner advisor.")
    parser.add_argument("--debug", action="store_true", help="Show thinking tokens, raw context, proposed_ingredients")
    args = parser.parse_args()
    asyncio.run(async_main(debug=args.debug))


if __name__ == "__main__":
    main()
