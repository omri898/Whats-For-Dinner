MODEL_NAME    = "qwen3-14b"
VLLM_BASE     = "http://localhost:8001/v1"
MAX_TURNS     = 30      # 10 full rotations — room for a real discussion
MIN_TURNS     = 12      # never exit before 4 full rotations
TURN_ORDER    = ["chef", "lazy", "nutricia"]
