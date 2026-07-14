#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

for model in qwen3 glm4 deepseek7b; do
  python tests/gpt5/phase417_native_generation_trace.py --model "$model"
done

python tests/gpt5/phase417_generation_physical_analysis.py
python -m unittest tests/gpt5/test_phase417_native_generation_physical_atlas.py
