#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ROUND_NAME="${ROUND_NAME:-shared_subspace_coupled_regime_analysis}"
MAX_ROLLOUT_CANDIDATES="${MAX_ROLLOUT_CANDIDATES:-5}"
PERTURB_SCALE="${PERTURB_SCALE:-0.35}"
ATTN_IMPLEMENTATIONS="${ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"

for MODEL in qwen3 glm4 deepseek7b; do
  python tests/gpt5/phase252_shared_subspace_coupled_regime_analysis.py \
    --model "${MODEL}" \
    --round-name "${ROUND_NAME}" \
    --max-rollout-candidates "${MAX_ROLLOUT_CANDIDATES}" \
    --perturb-scale "${PERTURB_SCALE}" \
    --attn-implementations "${ATTN_IMPLEMENTATIONS}"
done

python tests/gpt5/phase252_shared_subspace_coupled_regime_analysis.py \
  --round-name "${ROUND_NAME}" \
  --summarize
