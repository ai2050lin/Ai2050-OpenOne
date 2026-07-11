#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python tests/gpt5/phase342_copy_relay_execution_case_bank.py
for model in qwen3 glm4 deepseek7b; do
  python tests/gpt5/phase342_copy_relay_execution_invariance.py --model "$model"
done
python tests/gpt5/phase342_copy_relay_execution_analysis.py
