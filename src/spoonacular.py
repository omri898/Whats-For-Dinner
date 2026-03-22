"""
Spoonacular API client for recipe seeding.

Public API:
    fetch_recipes(cuisine, required_ingredients) -> list[dict]

The full raw API response is cached in `_recipe_cache` (module-level, reset per
process) and written to ./cache/spoonacular_cache.json on every successful call.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import httpx

from src.config import SPOONACULAR_API_KEY
from src.models import Cuisine

_ENDPOINT = "https://api.spoonacular.com/recipes/complexSearch"
_CACHE_FILE = Path("cache") / "spoonacular_cache.json"

# Module-level cache — holds the full raw response from the last successful call.
_recipe_cache: dict = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def _extract_instructions(recipe: dict) -> str:
    """Extract plain-text instructions from analyzedInstructions steps."""
    sections = recipe.get("analyzedInstructions") or []
    steps = []
    for section in sections:
        for s in section.get("steps", []):
            step_text = s.get("step", "").strip()
            if step_text:
                steps.append(f"{s['number']}. {step_text}")
    if steps:
        return " ".join(steps)
    # Fallback to plain instructions field (may be empty)
    return _strip_html(recipe.get("instructions", ""))


def _cuisine_param(cuisine: Cuisine) -> str | None:
    if cuisine == Cuisine.I_DONT_MIND:
        return None
    return cuisine.value.lower()


def _write_cache() -> None:
    try:
        _CACHE_FILE.parent.mkdir(exist_ok=True)
        _CACHE_FILE.write_text(json.dumps(_recipe_cache, indent=2))
    except Exception as exc:
        warnings.warn(f"[spoonacular] Failed to write cache: {exc}")


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------

async def fetch_recipes(
    cuisine: Cuisine,
    required_ingredients: list[str],
    n_fetch: int = 50,
    n_return: int = 5,
) -> list[dict]:
    """
    Query Spoonacular complexSearch and return the top `n_return` recipes by
    aggregateLikes.  Caches the full `n_fetch`-result raw response to disk.

    Returns [] on any error so Chef can gracefully fall back to free-form proposals.
    """
    params: dict = {
        "apiKey": SPOONACULAR_API_KEY,
        "addRecipeInformation": "true",
        "fillIngredients": "true",
        "addRecipeInstructions": "true",
        "number": n_fetch,
    }

    cuisine_str = _cuisine_param(cuisine)
    if cuisine_str:
        params["cuisine"] = cuisine_str

    if required_ingredients:
        params["includeIngredients"] = ",".join(required_ingredients)

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(_ENDPOINT, params=params)
            resp.raise_for_status()
    except Exception as exc:
        warnings.warn(
            f"[spoonacular] API call failed: {exc}. "
            "Chef will proceed without seeded recipes."
        )
        return []

    used = resp.headers.get("X-API-Quota-Used", "?")
    left = resp.headers.get("X-API-Quota-Left", "?")
    print(f"[spoonacular] quota used today: {used} | remaining today: {left}")

    data = resp.json()
    results = data.get("results", [])

    _recipe_cache["last_response"] = data
    _write_cache()

    if not results:
        warnings.warn(
            "[spoonacular] API returned 0 results. "
            "Chef will proceed without seeded recipes."
        )
        return []

    top = sorted(results, key=lambda r: r.get("aggregateLikes", 0), reverse=True)[:n_return]

    return [
        {
            "title": r.get("title", ""),
            "ingredients": [
                ing.get("name", "")
                for ing in r.get("extendedIngredients", [])
            ],
            "readyInMinutes": r.get("readyInMinutes"),
            "instructions": _extract_instructions(r),
            "aggregateLikes": r.get("aggregateLikes", 0),
        }
        for r in top
    ]
