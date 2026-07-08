#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-gate_product_protocol_trace}"
MAX_CASES="${MAX_CASES:-6}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${ROOT_DIR}"

for MODEL in qwen3 glm4 deepseek7b; do
  echo "== Phase240 ${MODEL} =="
  python tests/gpt5/phase240_gate_product_protocol_trace.py \
    --model "${MODEL}" \
    --round-name "${ROUND_NAME}" \
    --max-cases "${MAX_CASES}" \
    --max-new-tokens "${MAX_NEW_TOKENS}"
done

echo "== Phase240 summarize =="
python tests/gpt5/phase240_gate_product_protocol_trace.py \
  --summarize \
  --round-name "${ROUND_NAME}"
