#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-module_tree_gateup_causal_validation}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${ROOT_DIR}"

for MODEL in qwen3 glm4 deepseek7b; do
  echo "== Phase228 ${MODEL} =="
  python tests/gpt5/phase228_module_tree_gateup_causal_validation.py \
    --model "${MODEL}" \
    --round-name "${ROUND_NAME}"
done

echo "== Phase228 summarize =="
python tests/gpt5/phase228_module_tree_gateup_causal_validation.py \
  --summarize \
  --round-name "${ROUND_NAME}"
