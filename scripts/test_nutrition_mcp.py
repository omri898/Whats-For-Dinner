"""
Interactive tester for the mcp-opennutrition MCP server.

Usage:
    # Search for a food (returns top 3 results by default)
    python scripts/test_nutrition_mcp.py search spinach

    # Search and limit results
    python scripts/test_nutrition_mcp.py search spinach --page-size 1

    # Get full nutrition profile by ID (copy an id from a search result)
    python scripts/test_nutrition_mcp.py get fd_XwjLybfAIH0l

    # Search then auto-fetch the first result's full profile
    python scripts/test_nutrition_mcp.py search spinach --get-first

Requires NUTRITION_MCP_PATH to be set in .env or environment.

Available tools on the server:
  search-food-by-name  query=<str>  [pageSize=<int>]  [page=<int>]
  get-food-by-id       id=<fd_...>
  get-food-by-ean13    ean_13=<13-digit string>
  get-foods            [pageSize=<int>]  [page=<int>]   (browse all)
"""

from __future__ import annotations

import asyncio
import json
import sys
import argparse
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from src.config import NUTRITION_MCP_PATH


async def call_tool(tool_name: str, args: dict) -> None:
    server = StdioServerParameters(command="node", args=[NUTRITION_MCP_PATH])
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print(f"=== {tool_name} {args} ===\n")
            result = await session.call_tool(tool_name, args)
            for content in result.content:
                if hasattr(content, "text"):
                    try:
                        print(json.dumps(json.loads(content.text), indent=2))
                    except (json.JSONDecodeError, ValueError):
                        print(content.text)


async def search_then_get(query: str, page_size: int) -> None:
    server = StdioServerParameters(command="node", args=[NUTRITION_MCP_PATH])
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print(f"=== search-food-by-name: {query!r} (pageSize={page_size}) ===\n")
            result = await session.call_tool("search-food-by-name",
                                             {"query": query, "pageSize": page_size})
            text = result.content[0].text if result.content else "[]"
            data = json.loads(text)
            # server returns a list directly, or {"foods": [...]}
            foods = data if isinstance(data, list) else data.get("foods", [])
            if not foods:
                print("No results.")
                return

            first = foods[0]
            n100 = first.get("nutrition_100g", {})
            print(f"Top result: {first['name']}  (id: {first['id']})")
            print(f"  calories : {n100.get('calories')} kcal/100g")
            print(f"  protein  : {n100.get('protein')} g/100g")
            print(f"  total_fat: {n100.get('total_fat')} g/100g")
            print(f"  carbs    : {n100.get('carbohydrates')} g/100g\n")

            print(f"=== get-food-by-id: {first['id']} ===\n")
            result2 = await session.call_tool("get-food-by-id", {"id": first["id"]})
            text2 = result2.content[0].text if result2.content else "{}"
            print(json.dumps(json.loads(text2), indent=2))


def main() -> None:
    if not NUTRITION_MCP_PATH:
        print("Error: NUTRITION_MCP_PATH is not set. Add it to .env or export it.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Test the mcp-opennutrition MCP server")
    parser.add_argument("command", choices=["search", "get"],
                        help="'search' calls search-food-by-name; 'get' calls get-food-by-id")
    parser.add_argument("query", help="Search term or food ID (fd_...)")
    parser.add_argument("--page-size", type=int, default=3,
                        help="Number of results to return for search (default: 3)")
    parser.add_argument("--get-first", action="store_true",
                        help="After searching, also fetch full profile for the first result")
    args = parser.parse_args()

    if args.command == "search" and args.get_first:
        asyncio.run(search_then_get(args.query, args.page_size))
    elif args.command == "search":
        asyncio.run(call_tool("search-food-by-name",
                              {"query": args.query, "pageSize": args.page_size}))
    elif args.command == "get":
        asyncio.run(call_tool("get-food-by-id", {"id": args.query}))


if __name__ == "__main__":
    main()
