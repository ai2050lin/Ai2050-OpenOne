#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-reproducible_headset_validation}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

for MODEL in qwen3 glm4 deepseek7b; do
  echo "== Phase217 ${MODEL} =="
  python tests/gpt5/phase217_reproducible_headset_validation.py \
    --model "${MODEL}" \
    --round-name "${ROUND_NAME}" \
    --max-filter-rows 8 \
    --max-eval-rows 6 \
    --max-steps 10
done

python tests/gpt5/phase217_reproducible_headset_validation.py \
  --round-name "${ROUND_NAME}" \
  --summarize
