#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

ROUND_NAME="${1:-semantic_factor_causal_transfer_audit}"
COMMON_ARGS=(
  --round-name "${ROUND_NAME}"
  --phase937-round semantic_reuse_difference_state_atlas
  --relations category,color,function
  --max-objects-per-domain 6
  --templates-per-relation 2
  --min-train-per-label 2
  --alphas 0.5,1.0
  --batch-size 8
  --log-every 10
  --attn-implementations flash_attention_2,sdpa
)

for MODEL in qwen3 glm4 deepseek7b; do
  python tests/glm5/phase938_semantic_factor_causal_transfer_audit.py \
    --model "${MODEL}" \
    "${COMMON_ARGS[@]}"
done

python tests/glm5/phase938_semantic_factor_causal_transfer_audit.py \
  --summarize-round \
  --round-name "${ROUND_NAME}"
