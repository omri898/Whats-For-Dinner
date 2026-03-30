import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME    = "qwen3.5-27b"
VLLM_BASE     = "http://localhost:8001/v1"
MESSAGE_PAUSE_SECONDS = 1  # pause after each agent message (skipped in --debug mode)
MAX_TURNS     = 12      # max turns per round before falling back to best partial pick
MIN_TURNS     = 4       # min turns before agreement exit — ensures chef→lazy→nutricia→chef cycle
TURN_ORDER    = ["chef", "lazy", "nutricia"]

# Per-agent temperatures (1.0 = vLLM default). Tune here.
CHEF_TEMPERATURE     = 1.0
LAZY_TEMPERATURE     = 1.0
NUTRICIA_TEMPERATURE = 1.0

# Max tokens per LLM call. Prevents runaway generation (model generating endlessly
# after a tool return). Chef needs more headroom for full_instructions output.
CHEF_MAX_TOKENS      = 16384
LAZY_MAX_TOKENS      = 4096
NUTRICIA_MAX_TOKENS  = 8192

# Path to the mcp-opennutrition built entry point (dist/index.js after npm build).
# Set NUTRITION_MCP_PATH in .env or the environment before running.
NUTRITION_MCP_PATH = os.environ.get("NUTRITION_MCP_PATH", "")

# Debug: show raw tool returns (True) or the slimmed version sent to the model (False).
DEBUG_TOOL_RETURN_RAW = False

