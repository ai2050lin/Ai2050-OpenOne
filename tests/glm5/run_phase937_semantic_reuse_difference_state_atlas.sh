#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

ROUND_NAME="${1:-semantic_reuse_difference_state_atlas}"
COMMON_ARGS=(
  --round-name "${ROUND_NAME}"
  --relations category,color,function
  --max-objects-per-domain 6
  --templates-per-relation 2
  --batch-size 8
  --attn-implementations flash_attention_2,sdpa
)

for MODEL in qwen3 glm4 deepseek7b; do
  python tests/glm5/phase937_semantic_reuse_difference_state_atlas.py \
    --model "${MODEL}" \
    "${COMMON_ARGS[@]}"
done

python tests/glm5/phase937_semantic_reuse_difference_state_atlas.py \
  --summarize-round \
  --round-name "${ROUND_NAME}"
