#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python tests/gpt5/phase347_three_core_natural_trace_case_bank.py
for model in qwen3 glm4 deepseek7b; do
  python tests/gpt5/phase347_three_core_natural_trace.py --model "$model"
done
python tests/gpt5/phase347_three_core_natural_trace_analysis.py
