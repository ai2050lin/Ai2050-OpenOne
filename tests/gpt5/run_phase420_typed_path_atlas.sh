#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python tests/gpt5/phase420_typed_path_case_bank.py
for model in qwen3 glm4 deepseek7b; do
  python tests/gpt5/phase420_typed_path_trace.py --model "$model"
done
python tests/gpt5/phase420_typed_path_analysis.py
python -m unittest tests/gpt5/test_phase420_typed_path_atlas.py
