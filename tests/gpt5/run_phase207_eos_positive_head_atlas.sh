#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-eos_positive_head_atlas}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

for MODEL in qwen3 glm4 deepseek7b; do
  echo "== Phase207 ${MODEL} =="
  python tests/gpt5/phase207_eos_positive_head_atlas.py \
    --model "${MODEL}" \
    --round-name "${ROUND_NAME}" \
    --max-pairs 6 \
    --max-samples-per-pair 12 \
    --max-steps 32 \
    --max-states-per-class 24 \
    --batch-size 8 \
    --state-batch-size 4
done

python tests/gpt5/phase207_eos_positive_head_atlas.py \
  --round-name "${ROUND_NAME}" \
  --summarize
