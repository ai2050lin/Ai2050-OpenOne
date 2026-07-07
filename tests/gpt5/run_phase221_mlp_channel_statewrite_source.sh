#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-mlp_channel_statewrite_source}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${ROOT_DIR}"

for MODEL in qwen3 glm4 deepseek7b; do
  echo "== Phase221 ${MODEL} =="
  python tests/gpt5/phase221_mlp_channel_statewrite_source.py \
    --model "${MODEL}" \
    --round-name "${ROUND_NAME}"
done

echo "== Phase221 summarize =="
python tests/gpt5/phase221_mlp_channel_statewrite_source.py \
  --summarize \
  --round-name "${ROUND_NAME}"
