#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ROUND_NAME="${ROUND_NAME:-regime_level_causal_validation}"
MAX_CANDIDATES_PER_MODEL="${MAX_CANDIDATES_PER_MODEL:-10}"
MAX_ROLLOUT_TOKENS="${MAX_ROLLOUT_TOKENS:-4}"
PERTURB_SCALE="${PERTURB_SCALE:-0.35}"
ATTN_IMPLEMENTATIONS="${ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"

for MODEL in qwen3 glm4 deepseek7b; do
  python tests/gpt5/phase249_regime_level_causal_validation.py \
    --model "${MODEL}" \
    --round-name "${ROUND_NAME}" \
    --max-candidates-per-model "${MAX_CANDIDATES_PER_MODEL}" \
    --max-rollout-tokens "${MAX_ROLLOUT_TOKENS}" \
    --perturb-scale "${PERTURB_SCALE}" \
    --attn-implementations "${ATTN_IMPLEMENTATIONS}"
done

python tests/gpt5/phase249_regime_level_causal_validation.py \
  --round-name "${ROUND_NAME}" \
  --summarize
