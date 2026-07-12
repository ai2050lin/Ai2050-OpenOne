#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python tests/gpt5/phase351_signed_paired_trace_case_bank.py
for model in qwen3 glm4 deepseek7b; do
  python tests/gpt5/phase351_signed_paired_trace.py --model "$model"
done
python tests/gpt5/phase351_signed_paired_trace_analysis.py
