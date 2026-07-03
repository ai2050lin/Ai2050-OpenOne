#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-main}"
MAX_CASES="${2:-5}"
PROMPT_VARIANTS="${3:-natural_question,natural_category,classification}"
EDIT_MODES="${4:-zero,flip,half,scale_up}"
INCLUDE_SINGLE="${5:-1}"

cd "$(dirname "$0")/../.."

COMMON_ARGS=(
  --round-name "$ROUND_NAME"
  --max-cases-per-domain "$MAX_CASES"
  --prompt-variants "$PROMPT_VARIANTS"
  --edit-modes "$EDIT_MODES"
  --attn-implementations "flash_attention_2,sdpa"
  --max-new-tokens 8
  --topk-tokens 20
  --topk-blockers 10
)

if [[ "$INCLUDE_SINGLE" == "1" ]]; then
  COMMON_ARGS+=(--include-single-channels)
fi

for MODEL in qwen3 glm4 deepseek7b; do
  echo "===== Phase 862 ${ROUND_NAME}: ${MODEL} ====="
  python tests/glm5/phase862_negative_blocker_sign_mechanism_audit.py \
    --model "$MODEL" \
    "${COMMON_ARGS[@]}"
done

python tests/glm5/phase862_negative_blocker_sign_mechanism_audit.py \
  --round-name "$ROUND_NAME" \
  --summarize-round
