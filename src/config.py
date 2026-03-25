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

