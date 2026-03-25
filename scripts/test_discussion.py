"""Phase 4 test: sequential per-recipe rounds + recommendations."""

import asyncio

from src.agents import check_vllm
from src.discussion import run_all_rounds, display_recommendations
from src.models import Cuisine, LazyLevel


async def main() -> None:
    await check_vllm()

    picks = await run_all_rounds(
        cuisine=Cuisine.EUROPEAN,
        required_ingredients=["eggs", "feta cheese"],
        lazy_level=LazyLevel.MEDIUM,
    )

    display_recommendations(picks)

    print(f"\nPicks: {len(picks)}")
    for name in picks:
        print(f"  - {name}")


if __name__ == "__main__":
    asyncio.run(main())
