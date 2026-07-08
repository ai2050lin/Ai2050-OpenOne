#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-regime_level_direction_bank}"

for MODEL in qwen3 glm4 deepseek7b; do
  python tests/gpt5/phase248_regime_level_direction_bank.py \
    --model "${MODEL}" \
    --round-name "${ROUND_NAME}"
done

python tests/gpt5/phase248_regime_level_direction_bank.py \
  --round-name "${ROUND_NAME}" \
  --summarize
