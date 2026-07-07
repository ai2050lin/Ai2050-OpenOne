#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-source_restricted_value_ablation}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

for MODEL in qwen3 glm4 deepseek7b; do
  echo "== Phase218 ${MODEL} =="
  python tests/gpt5/phase218_source_restricted_value_ablation.py \
    --model "${MODEL}" \
    --round-name "${ROUND_NAME}" \
    --max-filter-rows 8 \
    --max-eval-rows 4 \
    --max-steps 8
done

python tests/gpt5/phase218_source_restricted_value_ablation.py \
  --round-name "${ROUND_NAME}" \
  --summarize
