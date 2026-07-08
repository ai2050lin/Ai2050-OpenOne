#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ROUND_NAME="${ROUND_NAME:-natural_regime_direction_extraction}"
CASES_PER_FAMILY="${CASES_PER_FAMILY:-2}"
ATTN_IMPLEMENTATIONS="${ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"

for MODEL in qwen3 glm4 deepseek7b; do
  python tests/gpt5/phase250_natural_regime_direction_extraction.py \
    --model "${MODEL}" \
    --round-name "${ROUND_NAME}" \
    --cases-per-family "${CASES_PER_FAMILY}" \
    --attn-implementations "${ATTN_IMPLEMENTATIONS}"
done

python tests/gpt5/phase250_natural_regime_direction_extraction.py \
  --round-name "${ROUND_NAME}" \
  --summarize
