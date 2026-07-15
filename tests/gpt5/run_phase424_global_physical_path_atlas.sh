#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export TOKENIZERS_PARALLELISM=false
export PROBE_ATTN_IMPLEMENTATION=eager

python tests/gpt5/phase424_global_physical_protocol.py --reuse-frozen
PROBE_TORCH_DTYPE=float16 python tests/gpt5/phase424_global_physical_collect.py --model qwen3
PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase424_global_physical_collect.py --model glm4
PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase424_global_physical_collect.py --model deepseek7b
python tests/gpt5/phase424_global_physical_analysis.py
python -m unittest tests/gpt5/test_phase424_global_physical_path_atlas.py
node tests/gpt5/phase415_multi_route_vis_source_contract.mjs
