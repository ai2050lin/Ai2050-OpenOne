#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="results/glm5_phase732_full_path_atlas_causal_edge_validation"
mkdir -p "$OUT_DIR"

COMMON_ARGS=(
  --top-heads "${PHASE732_TOP_HEADS:-4}"
  --max-new-tokens "${PHASE732_MAX_NEW_TOKENS:-4}"
  --hard-exit-after-model
)

if [[ -n "${PHASE732_MAX_PROMPT_PAIRS:-}" ]]; then
  COMMON_ARGS+=(--max-prompt-pairs "$PHASE732_MAX_PROMPT_PAIRS")
fi

if [[ -n "${PHASE732_MAX_HEAD_CASES:-}" ]]; then
  COMMON_ARGS+=(--max-head-cases "$PHASE732_MAX_HEAD_CASES")
fi

run_one() {
  local model="$1"
  echo "=== Phase732 ${model} start $(date '+%F %T') ===" | tee "$OUT_DIR/${model}.log"
  python tests/gpt5/phase732_full_path_atlas_causal_edge_validation.py \
    --model "$model" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee -a "$OUT_DIR/${model}.log"
  echo "=== Phase732 ${model} done $(date '+%F %T') ===" | tee -a "$OUT_DIR/${model}.log"
}

run_one qwen3
run_one glm4
run_one deepseek7b

python tests/gpt5/phase732_full_path_atlas_causal_edge_validation.py --summarize-only \
  2>&1 | tee "$OUT_DIR/summary.log"
