#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-pattern_running_contrast_atlas}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

for MODEL in qwen3 glm4 deepseek7b; do
  echo "== Phase209 ${MODEL} =="
  python tests/gpt5/phase209_pattern_running_contrast_atlas.py \
    --model "${MODEL}" \
    --round-name "${ROUND_NAME}" \
    --max-pairs 6 \
    --max-samples-per-pair 10 \
    --max-steps 20 \
    --batch-size 8
done

python tests/gpt5/phase209_pattern_running_contrast_atlas.py \
  --round-name "${ROUND_NAME}" \
  --summarize
