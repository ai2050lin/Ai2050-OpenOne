#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-route_head_causal_calibration}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

for MODEL in qwen3 glm4 deepseek7b; do
  echo "== Phase216 ${MODEL} =="
  python tests/gpt5/phase216_route_head_causal_calibration.py \
    --model "${MODEL}" \
    --round-name "${ROUND_NAME}" \
    --max-eval-rows 4 \
    --max-steps 10
done

python tests/gpt5/phase216_route_head_causal_calibration.py \
  --round-name "${ROUND_NAME}" \
  --summarize
