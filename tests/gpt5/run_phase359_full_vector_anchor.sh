#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python tests/gpt5/phase359_storage_budget.py
for model in qwen3 glm4 deepseek7b; do
  python tests/gpt5/phase359_full_vector_anchor.py --model "$model"
done
python tests/gpt5/phase359_full_vector_replay.py
