#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python tests/gpt5/phase338_block_causal_case_bank.py
for model in qwen3 glm4 deepseek7b; do
  python tests/gpt5/phase338_block_causal_screen.py --model "$model" --stage discovery
  python tests/gpt5/phase338_block_causal_screen.py --model "$model" --stage calibration
  python tests/gpt5/phase338_block_causal_screen.py --model "$model" --stage heldout
done
python tests/gpt5/phase338_block_causal_analysis.py
