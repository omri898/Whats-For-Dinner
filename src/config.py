import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME    = "qwen3-14b"
VLLM_BASE     = "http://localhost:8001/v1"
MESSAGE_PAUSE_SECONDS = 2  # pause after each agent message (skipped in --debug mode)
MAX_TURNS     = 12      # max turns per round before falling back to best partial pick
MIN_TURNS     = 4       # min turns per round — Chef must have at least one follow-up cycle
TURN_ORDER    = ["chef", "lazy", "nutricia"]

SPOONACULAR_API_KEY: str = os.environ.get("SPOONACULAR_API_KEY", "")
if not SPOONACULAR_API_KEY:
    raise ValueError(
        "SPOONACULAR_API_KEY is not set. Add it to your .env file or environment."
    )
