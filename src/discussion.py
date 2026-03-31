from __future__ import annotations

import datetime
import json
import sys
import time
import traceback
from pathlib import Path
from typing import IO

from rich.console import Console
from rich.panel import Panel

from pydantic_ai import Agent  # type: ignore
from pydantic_ai.messages import (  # type: ignore
    ThinkingPart,
    TextPart,
    SystemPromptPart,
    UserPromptPart,
    ToolCallPart,
    ToolReturnPart,
    PartStartEvent,
    PartDeltaEvent,
    TextPartDelta,
    ThinkingPartDelta,
)

from src.config import MAX_TURNS, MESSAGE_PAUSE_SECONDS, MIN_TURNS, TURN_ORDER, DEBUG_TOOL_RETURN_RAW
from src.agents import _slim_nutrition_response, _NUTRICIA_TOOLS
from src.models import (
    Cuisine,
    GroupMessage,
    LazyGroupContext,
    LazyLevel,
    RecipeCard,
)
from src.agents import chef_agent, lazy_agent, nutricia_agent

console = Console()

# ---------------------------------------------------------------------------
# Plain-text log file (set by setup_log() when --debug is active)
# ---------------------------------------------------------------------------

_log_file: IO[str] | None = None


def setup_log(path: Path) -> None:
    global _log_file
    _log_file = open(path, "w", encoding="utf-8")


def close_log() -> None:
    global _log_file
    if _log_file is not None:
        _log_file.close()
        _log_file = None


def _log(label: str, content: str) -> None:
    if _log_file is not None:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        _log_file.write(f"\n=== [{ts}] {label} ===\n{content}\n")
        _log_file.flush()


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
        _log("System", msg.text)
        return

    name, color = AGENT_DISPLAY.get(msg.agent, (msg.agent, "white"))
    approval_label = ""
    if msg.agent in ("lazy", "nutricia") and msg.approval is not None:
        approval_label = " · " + ("Approved" if msg.approval else "Declined")
    recipe_label = msg.recipe_card.recipe_name if msg.recipe_card else ""
    subtitle = (
        f"Turn {turn} · {msg.message_type} · → {msg.directed_at}"
        f"{' · ' + recipe_label if recipe_label else ''}"
        f"{approval_label}"
    )
    console.print(Panel(
        msg.text,
        title=f"[bold {color}]{name}[/bold {color}]",
        subtitle=f"[dim]{subtitle}[/dim]",
        border_style=color,
    ))
    _log(f"{name} | {subtitle}", msg.text)

    if debug and msg.recipe_card and msg.recipe_card.proposed_ingredients:
        console.print(f"  [dim]proposed_ingredients: {msg.recipe_card.proposed_ingredients}[/dim]")
        _log("Proposed Ingredients", str(msg.recipe_card.proposed_ingredients))


def _print_recipe_card(card: RecipeCard) -> None:
    """Print a full recipe card from a RecipeCard object."""
    lines: list[str] = []
    if card.estimated_time:
        lines.append(f"[bold]Time:[/bold] {card.estimated_time}")
    if card.proposed_ingredients:
        lines.append(f"[bold]Ingredients:[/bold] {', '.join(card.proposed_ingredients)}")
    instructions = card.full_instructions or card.cooking_summary
    if instructions:
        lines.append(f"\n[bold]Instructions:[/bold]\n{instructions}")
    if lines:
        console.print(Panel(
            "\n".join(lines),
            title=f"[bold green]Recipe Card: {card.recipe_name}[/bold green]",
            border_style="green",
            padding=(0, 1),
        ))


def _print_msg_parts(
    m: object,
    display_name: str,
    turn_num: int,
    *,
    skip_thinking_and_text: bool = False,
    log_only: bool = False,
) -> None:
    """Print all debug-relevant parts of a single ModelMessage as they stream in."""
    if not hasattr(m, "parts"):
        return
    for part in m.parts:  # type: ignore[attr-defined]
        if isinstance(part, SystemPromptPart):
            if not log_only:
                console.print(Panel(
                    part.content,
                    title="[bold cyan]System Prompt[/bold cyan]",
                    border_style="cyan",
                ))
            _log("System Prompt", part.content)
        elif isinstance(part, UserPromptPart):
            content = part.content if isinstance(part.content, str) else json.dumps(part.content, indent=2)
            if not log_only:
                console.print(Panel(
                    content,
                    title="[bold blue]User Prompt[/bold blue]",
                    border_style="blue",
                ))
            _log("User Prompt", content)
        elif isinstance(part, ThinkingPart):
            if not skip_thinking_and_text and not log_only:
                console.print(Panel(
                    part.content,
                    title="[bold magenta]Thinking / Reasoning[/bold magenta]",
                    border_style="magenta",
                ))
            _log("Thinking", part.content)
        elif isinstance(part, ToolCallPart):
            args_str = part.args if isinstance(part.args, str) else json.dumps(part.args, indent=2)
            if not log_only:
                console.print(Panel(
                    f"tool: {part.tool_name}\nargs: {args_str}",
                    title="[bold yellow]Tool Call[/bold yellow]",
                    border_style="yellow",
                ))
            _log("Tool Call", f"tool: {part.tool_name}\nargs: {args_str}")
        elif isinstance(part, ToolReturnPart):
            content_str = str(part.content)
            is_nutrition = part.tool_name in _NUTRICIA_TOOLS
            if is_nutrition and not DEBUG_TOOL_RETURN_RAW:
                # part.content is a Python object — serialize to JSON so
                # _slim_nutrition_response can parse and strip fields.
                try:
                    json_str = json.dumps(part.content)
                except (TypeError, ValueError):
                    json_str = content_str
                display_str = _slim_nutrition_response(json_str)
                label = f"[bold yellow]Tool Return (slimmed) · {part.tool_name}[/bold yellow]"
                log_label = f"Tool Return (slimmed): {part.tool_name}"
            else:
                display_str = content_str
                label = f"[bold yellow]Tool Return · {part.tool_name}[/bold yellow]"
                log_label = f"Tool Return: {part.tool_name}"
            if not log_only:
                console.print(Panel(
                    display_str,
                    title=label,
                    border_style="yellow",
                ))
            _log(log_label, display_str)
        elif isinstance(part, TextPart) and part.content.strip():
            if not skip_thinking_and_text and not log_only:
                console.print(Panel(
                    part.content,
                    title="[bold white]Raw LLM Text[/bold white]",
                    border_style="white",
                ))
            _log("Raw LLM Text", part.content)


def _round_agreement(
    history: list[GroupMessage], current_card: RecipeCard | None
) -> str | None:
    """Return the agreed recipe name (from current_card) if both Lazy and Nutricia
    approved after the most recent chef proposal/pivot, else None.

    Uses current_card as the canonical name — no recipe_name matching required on
    Lazy/Nutricia messages, preventing hallucinated-name mismatches.
    Only approvals that arrive after the last proposal/pivot are counted, so a
    previous approval for an earlier recipe cannot carry over to a new proposal.
    """
    if current_card is None:
        return None

    # Find the index of the most recent chef proposal or pivot.
    last_proposal_idx = next(
        (i for i in range(len(history) - 1, -1, -1)
         if history[i].agent == "chef" and history[i].message_type in ("proposal", "pivot")),
        None,
    )
    if last_proposal_idx is None:
        return None

    post_proposal = history[last_proposal_idx + 1:]
    last_lazy = next((m for m in reversed(post_proposal) if m.agent == "lazy"), None)
    last_nutricia = next((m for m in reversed(post_proposal) if m.agent == "nutricia"), None)
    if not last_lazy or not last_nutricia:
        return None
    if last_lazy.approval is True and last_nutricia.approval is True:
        return current_card.recipe_name
    return None


def pick_best_from_round(history: list[GroupMessage]) -> str | None:
    """
    Fallback: return the recipe from this round's history with the most approvals.
    Full consensus (both approved) beats partial (one approved).
    Tracks the current recipe from Chef proposals and attributes reactor
    approvals to it (rather than relying on reactor-set recipe names).
    Returns None if no recipes were proposed.
    """
    order: list[str] = []
    approvals: dict[str, dict[str, bool]] = {}
    name_map: dict[str, str] = {}
    current_key: str | None = None

    for msg in history:
        if msg.recipe_card and msg.message_type in ("proposal", "pivot"):
            current_key = msg.recipe_card.recipe_name.lower()
            if current_key not in approvals:
                order.append(current_key)
                approvals[current_key] = {}
            name_map.setdefault(current_key, msg.recipe_card.recipe_name)
        if msg.agent in ("lazy", "nutricia") and msg.approval is not None and current_key:
            if current_key not in approvals:
                order.append(current_key)
                approvals[current_key] = {}
            approvals[current_key][msg.agent] = msg.approval
            name_map.setdefault(current_key, current_key)

    if not order:
        return None

    def score(key: str) -> int:
        return sum(1 for v in approvals.get(key, {}).values() if v)

    best = max(order, key=score)
    return name_map.get(best, best)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_recipe_search_results(content: str) -> list[dict]:
    """Parse recipe_search Markdown output into {title, url} dicts.

    The MCP server returns plain Markdown text, e.g.:
        **Veggie Teriyaki Stir-Fry**
        by Cookie and Kate
        Source: blogs
        https://cookieandkate.com/...
    """
    results = []
    lines = content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("**") and line.endswith("**") and len(line) > 4:
            title = line[2:-2].strip()
            for j in range(i + 1, min(i + 5, len(lines))):
                candidate = lines[j].strip()
                if candidate.startswith("http"):
                    results.append({"title": title, "url": candidate})
                    i = j
                    break
        i += 1
    return results


# ---------------------------------------------------------------------------
# Per-recipe round
# ---------------------------------------------------------------------------

async def run_round(
    context: LazyGroupContext,
    *,
    round_num: int,
    debug: bool = False,
) -> tuple[str | None, RecipeCard | None, list[GroupMessage]]:
    """
    Run one mini-discussion targeting a single recipe.
    Exits immediately when Lazy and Nutricia both approve the same recipe.
    Falls back to pick_best_from_round on MAX_TURNS exhaustion.
    Returns (recipe_name_or_none, round_history).
    """
    console.print()
    console.rule(f"[bold blue]Round {round_num}[/bold blue]")
    console.print()
    _log("Round", str(round_num))

    # Always log full context at round start for crash diagnosis
    ctx_json = context.model_dump_json(indent=2)
    _log("Context at round start", ctx_json)
    if debug:
        console.print("[dim]Context at round start:[/dim]")
        console.print(f"[dim]{ctx_json}[/dim]")

    agreed: str | None = None
    context.current_recipe_card = None

    for turn_num in range(1, MAX_TURNS + 1):
        agent_name = TURN_ORDER[(turn_num - 1) % len(TURN_ORDER)]
        agent = AGENT_MAP[agent_name]
        display_name = AGENT_DISPLAY[agent_name][0]

        # Always log pre-call context (even in non-debug mode) for crash diagnosis
        _log("Turn start", (
            f"{display_name} (turn {turn_num}) | "
            f"history={len(context.history)} msgs | "
            f"recipe_card={context.current_recipe_card.recipe_name if context.current_recipe_card else 'none'} | "
            f"search_cache={len(context.search_results_this_round)} hits"
        ))

        if debug:
            # Stream via iter() so debug output appears even if the run crashes.
            console.print(f"\n[dim bold]── DEBUG: {display_name} (turn {turn_num}) ──[/dim bold]")
            _log("Turn", f"{display_name} (turn {turn_num})")
            seen = 0
            async with agent.iter("Your turn.", deps=context) as agent_run:
                try:
                    async for node in agent_run:
                        if Agent.is_model_request_node(node):
                            # Stream tokens live as they arrive
                            did_stream = False
                            async with node.stream(agent_run.ctx) as stream:
                                async for event in stream:
                                    if isinstance(event, PartStartEvent):
                                        if isinstance(event.part, ThinkingPart):
                                            console.print("\n[bold magenta]◆ Thinking[/bold magenta]")
                                            did_stream = True
                                        elif isinstance(event.part, TextPart):
                                            console.print("\n[bold white]◆ Generating[/bold white]")
                                            did_stream = True
                                    elif isinstance(event, PartDeltaEvent):
                                        if isinstance(event.delta, ThinkingPartDelta) and event.delta.content_delta:
                                            sys.stdout.write(event.delta.content_delta)
                                            sys.stdout.flush()
                                        elif isinstance(event.delta, TextPartDelta):
                                            sys.stdout.write(event.delta.content_delta)
                                            sys.stdout.flush()
                            if did_stream:
                                sys.stdout.write("\n")
                                sys.stdout.flush()
                            # Print tool calls etc.; skip thinking/text (shown live above)
                            all_msgs = agent_run.all_messages()
                            for m in all_msgs[seen:]:
                                _print_msg_parts(m, display_name, turn_num, skip_thinking_and_text=did_stream)
                            seen = len(all_msgs)
                        else:
                            all_msgs = agent_run.all_messages()
                            for m in all_msgs[seen:]:
                                _print_msg_parts(m, display_name, turn_num)
                            seen = len(all_msgs)
                except Exception as exc:
                    console.print(f"\n[bold red]Agent crashed: {exc}[/bold red]")
                    _log("Agent crashed", f"{exc}\n\n{traceback.format_exc()}")
                    for m in agent_run.all_messages()[seen:]:
                        _print_msg_parts(m, display_name, turn_num)
                    _log("Context at crash", context.model_dump_json(indent=2))
                    raise

            result = agent_run.result
            structured_json = result.output.model_dump_json(indent=2)
            console.print(Panel(
                structured_json,
                title="[bold green]Structured Output (GroupMessage)[/bold green]",
                border_style="green",
            ))
            _log("Structured Output (GroupMessage)", structured_json)
            console.print("[dim]── Press Enter for next agent ──[/dim]", end=" ")
            input()
        else:
            # Use iter() instead of run() so we can capture partial messages on crash
            async with agent.iter("Your turn.", deps=context) as agent_run:
                try:
                    with console.status(f"[dim]{display_name} is thinking...[/dim]"):
                        async for _node in agent_run:
                            pass
                        time.sleep(MESSAGE_PAUSE_SECONDS)
                except Exception as exc:
                    # Log everything the model produced before the crash
                    _log("Agent crashed", f"{display_name} (turn {turn_num})\n{exc}")
                    for m in agent_run.all_messages():
                        _print_msg_parts(m, display_name, turn_num, log_only=True)
                    _log("Context at crash", context.model_dump_json(indent=2))
                    raise
            result = agent_run.result
            # Log full turn details post-hoc (same content as debug path, no console output)
            for m in result.all_messages():
                _print_msg_parts(m, display_name, turn_num, skip_thinking_and_text=True, log_only=True)
            _log("Structured Output (GroupMessage)", result.output.model_dump_json(indent=2))

        msg = result.output
        context.history.append(msg)
        _print_message(msg, turn_num, debug=debug)

        # Update canonical recipe card whenever Chef proposes or pivots
        if msg.agent == "chef" and msg.message_type in ("proposal", "pivot") and msg.recipe_card:
            context.current_recipe_card = msg.recipe_card

        # Capture recipe_search query + results on Chef's first search this round
        if agent_name == "chef" and not context.search_results_this_round:
            for m in result.all_messages():
                if not hasattr(m, "parts"):
                    continue
                for part in m.parts:
                    if isinstance(part, ToolCallPart) and part.tool_name == "recipe_search":
                        try:
                            context.search_query_this_round = json.loads(part.args_as_json_str()).get("query", "")
                        except Exception:
                            pass
                    if isinstance(part, ToolReturnPart) and part.tool_name == "recipe_search":
                        hits = _parse_recipe_search_results(str(part.content))
                        if hits:
                            context.search_results_this_round = hits

        # Exit on agreement after MIN_TURNS (chef→lazy→nutricia→chef), checked every turn
        if turn_num >= MIN_TURNS:
            agreed = _round_agreement(context.history, context.current_recipe_card)
            if agreed:
                console.print()
                console.print(f"[bold green]Agreement: {agreed}[/bold green]")
                _log("Agreement", agreed)
                if context.current_recipe_card:
                    _print_recipe_card(context.current_recipe_card)
                break
    else:
        console.print()
        console.print("[bold yellow]MAX_TURNS reached — picking best candidate.[/bold yellow]")
        agreed = pick_best_from_round(context.history)
        if agreed:
            _log("Fallback pick", agreed)
            if context.current_recipe_card:
                _print_recipe_card(context.current_recipe_card)

    round_history = list(context.history)
    return agreed, context.current_recipe_card, round_history


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
) -> list[RecipeCard]:
    """
    Run num_rounds sequential per-recipe discussions.
    After each round, agreed recipes are passed forward so Chef avoids similar proposals.
    Returns list of up to num_rounds RecipeCard objects for the agreed recipes.
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

    picks: list[RecipeCard] = []

    async with chef_agent, nutricia_agent:
        for round_num in range(1, num_rounds + 1):
            context.history = []
            context.search_results_this_round = []
            context.search_query_this_round = ""
            context.agreed_recipes = [c.recipe_name for c in picks]

            pick_name, pick_card, _ = await run_round(context, round_num=round_num, debug=debug)
            if pick_name and pick_card:
                picks.append(pick_card)

    return picks


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display_recommendations(picks: list[RecipeCard]) -> None:
    console.rule("[bold blue]Recommendations[/bold blue]")
    console.print()
    for i, card in enumerate(picks, 1):
        console.print(f"  [bold]{i}. {card.recipe_name}[/bold]")
        console.print()
        _print_recipe_card(card)
    console.print()
