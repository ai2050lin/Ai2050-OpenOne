#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

for model in qwen3 glm4 deepseek7b; do
  python tests/gpt5/phase352_generated_time_trace.py --model "$model"
done
python tests/gpt5/phase352_generated_time_trace_analysis.py
