run:
	conda run --no-capture-output -n dinner python -m src.main

run-debug:
	conda run --no-capture-output -n dinner python -m src.main --debug

notebook:
	jupyter notebook notebook.ipynb

test-chef:
	conda run --no-capture-output -n dinner env PYTHONPATH=. python scripts/test_chef.py

test-round:
	conda run --no-capture-output -n dinner env PYTHONPATH=. python scripts/test_one_round.py

test-discussion:
	conda run --no-capture-output -n dinner env PYTHONPATH=. python scripts/test_discussion.py

check-vllm:
	conda run --no-capture-output -n dinner python -c "import asyncio; from src.agents import check_vllm; asyncio.run(check_vllm())"

check-mcp:
	@command -v recipe-mcp-server >/dev/null 2>&1 \
		&& echo "recipe-mcp-server found: $$(which recipe-mcp-server)" \
		|| (echo "recipe-mcp-server not found. Install with: npm install -g recipe-mcp-server" && exit 1)

check-nutrition-mcp:
	conda run --no-capture-output -n dinner env PYTHONPATH=. python -c "\
import os, sys; \
from src.config import NUTRITION_MCP_PATH; \
p = NUTRITION_MCP_PATH; \
sys.exit(print('ERROR: NUTRITION_MCP_PATH is not set in .env') or 1) if not p else \
sys.exit(print(f'ERROR: File not found: {p}') or 1) if not os.path.isfile(p) else \
print(f'mcp-opennutrition found: {p}')"

start-vllm-2-4090:
	rm -rf ~/.cache/vllm/torch_compile_cache
	conda run --no-capture-output -n dinner python -m vllm.entrypoints.openai.api_server \
		--model Qwen/Qwen3.5-27B-GPTQ-Int4 \
		--served-model-name qwen3.5-27b \
		--quantization gptq_marlin \
		--max-model-len 32768 \
		--port 8001 \
		--tool-call-parser qwen3_xml \
		--enable-auto-tool-choice \
		--reasoning-parser qwen3 \
		--override-generation-config '{"stop_token_ids": [151645, 151643], "top_k": 20, "top_p": 0.8}' \
		--tensor-parallel-size 2 \
		--max-cudagraph-capture-size 128

start-vllm-6000:
	rm -rf ~/.cache/vllm/torch_compile_cache
	conda run --no-capture-output -n dinner python -m vllm.entrypoints.openai.api_server \
		--model Qwen/Qwen3.5-27B-GPTQ-Int4 \
		--served-model-name qwen3.5-27b \
		--quantization gptq_marlin \
		--port 8001 \
		--tool-call-parser qwen3_xml \
		--enable-auto-tool-choice \
		--reasoning-parser qwen3 \
		--override-generation-config '{"stop_token_ids": [151645, 151643], "top_k": 20, "top_p": 0.8}'
