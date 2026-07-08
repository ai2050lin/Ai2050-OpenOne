#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ROUND_NAME="${ROUND_NAME:-closure_candidate_stop_validation}"
MAX_CANDIDATES_PER_MODEL="${MAX_CANDIDATES_PER_MODEL:-15}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
PERTURB_SCALE="${PERTURB_SCALE:-0.35}"
ATTN_IMPLEMENTATIONS="${ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"

for MODEL in qwen3 glm4 deepseek7b; do
  python tests/gpt5/phase254_closure_candidate_stop_validation.py \
    --model "${MODEL}" \
    --round-name "${ROUND_NAME}" \
    --max-candidates-per-model "${MAX_CANDIDATES_PER_MODEL}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --perturb-scale "${PERTURB_SCALE}" \
    --attn-implementations "${ATTN_IMPLEMENTATIONS}"
done

python tests/gpt5/phase254_closure_candidate_stop_validation.py \
  --round-name "${ROUND_NAME}" \
  --summarize
