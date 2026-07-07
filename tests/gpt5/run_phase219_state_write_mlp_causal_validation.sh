#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-state_write_mlp_causal_validation}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${ROOT_DIR}"

for MODEL in qwen3 glm4 deepseek7b; do
  echo "== Phase219 ${MODEL} =="
  python tests/gpt5/phase219_state_write_mlp_causal_validation.py \
    --model "${MODEL}" \
    --round-name "${ROUND_NAME}"
done

echo "== Phase219 summarize =="
python tests/gpt5/phase219_state_write_mlp_causal_validation.py \
  --summarize \
  --round-name "${ROUND_NAME}"
