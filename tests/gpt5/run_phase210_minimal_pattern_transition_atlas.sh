#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-minimal_pattern_transition_atlas}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

for MODEL in qwen3 glm4 deepseek7b; do
  echo "== Phase210 ${MODEL} =="
  python tests/gpt5/phase210_minimal_pattern_transition_atlas.py \
    --model "${MODEL}" \
    --round-name "${ROUND_NAME}" \
    --max-pairs 5 \
    --max-samples-per-pair 8 \
    --max-steps 12 \
    --batch-size 4
done

python tests/gpt5/phase210_minimal_pattern_transition_atlas.py \
  --round-name "${ROUND_NAME}" \
  --summarize
