#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python tests/gpt5/phase339_cross_task_boundary_case_bank.py
for model in qwen3 glm4 deepseek7b; do
  python tests/gpt5/phase339_cross_task_boundary_audit.py --model "$model"
done
python tests/gpt5/phase339_cross_task_boundary_analysis.py
