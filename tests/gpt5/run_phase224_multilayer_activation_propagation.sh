#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-multilayer_activation_propagation}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${ROOT_DIR}"

for MODEL in qwen3 glm4 deepseek7b; do
  echo "== Phase224 ${MODEL} =="
  python tests/gpt5/phase224_multilayer_activation_propagation.py \
    --model "${MODEL}" \
    --round-name "${ROUND_NAME}"
done

echo "== Phase224 summarize =="
python tests/gpt5/phase224_multilayer_activation_propagation.py \
  --summarize \
  --round-name "${ROUND_NAME}"
