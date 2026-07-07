#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-readout_competition_threshold}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${ROOT_DIR}"

for MODEL in qwen3 glm4 deepseek7b; do
  echo "== Phase225 ${MODEL} =="
  python tests/gpt5/phase225_readout_competition_threshold.py \
    --model "${MODEL}" \
    --round-name "${ROUND_NAME}"
done

echo "== Phase225 summarize =="
python tests/gpt5/phase225_readout_competition_threshold.py \
  --summarize \
  --round-name "${ROUND_NAME}"
