"""Phase 4 test: full discussion loop + supervisor."""

import asyncio

from src.agents import check_vllm
from src.discussion import run_discussion, run_supervisor
from src.models import Cuisine, LazyLevel


async def main() -> None:
    await check_vllm()

    history = await run_discussion(
        cuisine=Cuisine.ITALIAN,
        required_ingredients=["eggs", "feta cheese"],
        lazy_level=LazyLevel.MEDIUM,
    )

    recommendations = await run_supervisor(history)

    print(f"\nTotal turns: {len([m for m in history if m.agent != 'system'])}")
    print(f"Picks: {len(recommendations.picks)}")
    for pick in recommendations.picks:
        print(f"  - {pick.recipe_name}: {pick.why_it_won}")


if __name__ == "__main__":
    asyncio.run(main())
