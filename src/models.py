from __future__ import annotations
from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field


class LazyLevel(str, Enum):
    AMBITIOUS = "Feeling Ambitious"
    MEDIUM    = "I Guess I'll Cook"
    COUCH     = "Don't Make Me Move"

    def __str__(self) -> str:
        return self.value


class Cuisine(str, Enum):
    AFRICAN        = "African"
    AMERICAN       = "American"
    ASIAN          = "Asian"
    EUROPEAN       = "European"
    LATIN_AMERICAN = "Latin American"
    MIDDLE_EASTERN = "Middle Eastern"
    MEDITERRANEAN  = "Mediterranean"
    ANYTHING       = "Any cuisine"

    def __str__(self) -> str:
        return self.value


MessageType = Literal["proposal", "reaction", "pivot", "defense", "concession", "system"]


class GroupMessage(BaseModel):
    agent: Literal["chef", "lazy", "nutricia", "system"]
    directed_at: Literal["chef", "lazy", "nutricia", "all"] = "all"
    message_type: MessageType
    text: str
    # Only set on proposal/pivot turns:
    recipe_name: str | None = None
    proposed_ingredients: list[str] | None = None  # ingredients in the proposed recipe
    estimated_time: str | None = None              # e.g. "35 minutes"
    cooking_summary: str | None = None             # 2-4 sentence how-to summary
    full_instructions: str | None = None           # complete step-by-step from recipe_get, for end-of-round display
    # Only set on reaction/concession turns (chef never sets this):
    approval: bool | None = None


class GroupContext(BaseModel):
    """
    Shared discussion context.
    - Chef sees all fields.
    - Lazy sees required_ingredients and proposed_ingredients from history
      (NOT available_ingredients).
    - Nutricia sees proposed_ingredients from history
      (NOT available_ingredients, NOT required_ingredients directly).
    lazy_level is intentionally absent here; only LazyGroupContext carries it.
    """
    user_request: str
    cuisine: Cuisine = Cuisine.ANYTHING
    required_ingredients: list[str] = Field(
        default_factory=list,
        description="Ingredients the user insists must appear in every recipe. Non-negotiable from the start.",
    )
    available_ingredients: list[str] = Field(
        default_factory=list,
        description="Pantry items loaded from data/ingredients.json. Chef's domain only.",
    )
    agreed_recipes: list[str] = Field(
        default_factory=list,
        description="Names of recipes agreed upon in prior rounds. Empty in round 1. Used by Chef to avoid similar proposals.",
    )
    search_results_this_round: list[dict] = Field(
        default_factory=list,
        description="recipe_search hits from Chef's first search this round. Cleared between rounds. Injected into Chef's prompt for pivot turns.",
    )
    history: list[GroupMessage] = Field(default_factory=list)


class LazyGroupContext(GroupContext):
    """
    Full context — passed ONLY to The Lazy Advisor.
    lazy_level must never appear in a GroupContext handed to Chef or Nutricia.
    """
    lazy_level: LazyLevel = LazyLevel.MEDIUM


