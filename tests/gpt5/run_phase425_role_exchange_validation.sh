#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export TOKENIZERS_PARALLELISM=false
export PROBE_ATTN_IMPLEMENTATION=eager

python tests/gpt5/phase425_role_exchange_protocol.py --reuse-frozen
PROBE_TORCH_DTYPE=float16 python tests/gpt5/phase425_role_exchange_collect.py --model qwen3 --stage open
PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase425_role_exchange_collect.py --model glm4 --stage open
PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase425_role_exchange_collect.py --model deepseek7b --stage open
python tests/gpt5/phase425_role_exchange_analysis.py --stage preseal
python tests/gpt5/phase425_role_exchange_posthoc_audit.py
python -m unittest tests/gpt5/test_phase425_role_exchange_validation.py
node tests/gpt5/phase415_multi_route_vis_source_contract.mjs
