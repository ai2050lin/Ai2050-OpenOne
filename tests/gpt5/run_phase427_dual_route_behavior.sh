#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export TOKENIZERS_PARALLELISM=false
export PROBE_ATTN_IMPLEMENTATION=eager

python tests/gpt5/phase427_dual_route_protocol.py --reuse-frozen

PROBE_TORCH_DTYPE=float16 python tests/gpt5/phase427_behavior_collect.py --model qwen3 --stage instrument
PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase427_behavior_collect.py --model glm4 --stage instrument
PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase427_behavior_collect.py --model deepseek7b --stage instrument
python tests/gpt5/phase427_behavior_analysis.py --stage instrument

PROBE_TORCH_DTYPE=float16 python tests/gpt5/phase427_behavior_collect.py --model qwen3 --stage open
PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase427_behavior_collect.py --model glm4 --stage open
PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase427_behavior_collect.py --model deepseek7b --stage open
python tests/gpt5/phase427_behavior_analysis.py --stage open

if python -c 'import json; from pathlib import Path; raise SystemExit(0 if json.loads(Path("tests/gpt5/result/phase427_dual_route_behavior/phase427_open_gate_freeze.json").read_text())["sealed_behavior_unlock"] else 1)'
then
  PROBE_TORCH_DTYPE=float16 python tests/gpt5/phase427_behavior_collect.py --model qwen3 --stage sealed
  PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase427_behavior_collect.py --model glm4 --stage sealed
  PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase427_behavior_collect.py --model deepseek7b --stage sealed
  python tests/gpt5/phase427_behavior_analysis.py --stage sealed
fi

python -m unittest tests/gpt5/test_phase427_dual_route_behavior.py
node tests/gpt5/phase415_multi_route_vis_source_contract.mjs
