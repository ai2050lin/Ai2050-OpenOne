#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-channel_activation_gate_validation}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${ROOT_DIR}"

for MODEL in qwen3 glm4 deepseek7b; do
  echo "== Phase223 ${MODEL} =="
  python tests/gpt5/phase223_channel_activation_gate_validation.py \
    --model "${MODEL}" \
    --round-name "${ROUND_NAME}"
done

echo "== Phase223 summarize =="
python tests/gpt5/phase223_channel_activation_gate_validation.py \
  --summarize \
  --round-name "${ROUND_NAME}"
