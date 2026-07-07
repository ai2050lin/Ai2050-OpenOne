#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-natural_trigger_channel_activation}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${ROOT_DIR}"

for MODEL in qwen3 glm4 deepseek7b; do
  echo "== Phase226 ${MODEL} =="
  python tests/gpt5/phase226_natural_trigger_channel_activation.py \
    --model "${MODEL}" \
    --round-name "${ROUND_NAME}"
done

echo "== Phase226 summarize =="
python tests/gpt5/phase226_natural_trigger_channel_activation.py \
  --summarize \
  --round-name "${ROUND_NAME}"
