"""Phase 4 test: full discussion loop + recommendations."""

import asyncio

from src.agents import check_vllm
from src.discussion import run_discussion, pick_recipes, display_recommendations
from src.models import Cuisine, LazyLevel


async def main() -> None:
    await check_vllm()

    history = await run_discussion(
        cuisine=Cuisine.ITALIAN,
        required_ingredients=["eggs", "feta cheese"],
        lazy_level=LazyLevel.MEDIUM,
    )

    picks = pick_recipes(history)
    display_recommendations(picks)

    print(f"\nTotal turns: {len([m for m in history if m.agent != 'system'])}")
    print(f"Picks: {len(picks)}")
    for name in picks:
        print(f"  - {name}")


if __name__ == "__main__":
    asyncio.run(main())
