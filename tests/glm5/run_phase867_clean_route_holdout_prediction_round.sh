#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-main}"
MAX_CASES="${2:-4}"
PROMPT_VARIANTS="${3:-holdout_category_short,holdout_kind_phrase,holdout_label}"
INCLUDE_CONTROLS="${4:-1}"

cd "$(dirname "$0")/../.."

COMMON_ARGS=(
  --round-name "$ROUND_NAME"
  --max-cases-per-domain "$MAX_CASES"
  --prompt-variants "$PROMPT_VARIANTS"
  --attn-implementations "flash_attention_2,sdpa"
  --max-new-tokens 8
  --topk-tokens 20
  --topk-blockers 10
  --object-delta-threshold 0.25
)

if [[ "$INCLUDE_CONTROLS" == "1" ]]; then
  COMMON_ARGS+=(--include-non-clean-controls)
fi

for MODEL in qwen3 glm4 deepseek7b; do
  echo "===== Phase 867 ${ROUND_NAME}: ${MODEL} ====="
  python tests/glm5/phase867_clean_route_holdout_prediction.py \
    --model "$MODEL" \
    "${COMMON_ARGS[@]}"
done

python tests/glm5/phase867_clean_route_holdout_prediction.py \
  --round-name "$ROUND_NAME" \
  --summarize-round
