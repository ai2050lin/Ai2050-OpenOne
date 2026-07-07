#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-window_direction_prompt_trigger}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

for MODEL in qwen3 glm4 deepseek7b; do
  echo "== Phase213 ${MODEL} =="
  python tests/gpt5/phase213_window_direction_prompt_trigger.py \
    --model "${MODEL}" \
    --round-name "${ROUND_NAME}" \
    --max-rows-per-group 8 \
    --max-donor-rows 6 \
    --max-eval-rows 6 \
    --max-steps 12 \
    --direction-scale 0.7
done

python tests/gpt5/phase213_window_direction_prompt_trigger.py \
  --round-name "${ROUND_NAME}" \
  --summarize
