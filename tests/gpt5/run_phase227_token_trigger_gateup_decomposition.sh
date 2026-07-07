#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-token_trigger_gateup_decomposition}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${ROOT_DIR}"

for MODEL in qwen3 glm4 deepseek7b; do
  echo "== Phase227 ${MODEL} =="
  python tests/gpt5/phase227_token_trigger_gateup_decomposition.py \
    --model "${MODEL}" \
    --round-name "${ROUND_NAME}"
done

echo "== Phase227 summarize =="
python tests/gpt5/phase227_token_trigger_gateup_decomposition.py \
  --summarize \
  --round-name "${ROUND_NAME}"
