#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-statewrite_factor_competition}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${ROOT_DIR}"

for MODEL in qwen3 glm4 deepseek7b; do
  echo "== Phase222 ${MODEL} =="
  python tests/gpt5/phase222_statewrite_factor_competition.py \
    --model "${MODEL}" \
    --round-name "${ROUND_NAME}"
done

echo "== Phase222 summarize =="
python tests/gpt5/phase222_statewrite_factor_competition.py \
  --summarize \
  --round-name "${ROUND_NAME}"
