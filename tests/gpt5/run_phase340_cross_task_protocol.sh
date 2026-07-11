#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python tests/gpt5/phase340_cross_task_protocol_case_bank.py
for model in qwen3 glm4 deepseek7b; do
  python tests/gpt5/phase340_cross_task_protocol_qualification.py --model "$model"
done
python tests/gpt5/phase340_batch_invariance_diagnostic.py --model glm4
python tests/gpt5/phase340_cross_task_protocol_analysis.py
