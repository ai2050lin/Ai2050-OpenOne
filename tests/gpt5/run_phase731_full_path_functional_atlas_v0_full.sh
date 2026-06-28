#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="results/glm5_phase731_full_path_functional_atlas_v0"
mkdir -p "$OUT_DIR"

COMMON_ARGS=(
  --top-heads "${PHASE731_TOP_HEADS:-4}"
  --max-new-tokens "${PHASE731_MAX_NEW_TOKENS:-4}"
  --log-every "${PHASE731_LOG_EVERY:-8}"
  --hard-exit-after-model
)

if [[ -n "${PHASE731_MAX_CASES:-}" ]]; then
  COMMON_ARGS+=(--max-cases "$PHASE731_MAX_CASES")
fi

run_one() {
  local model="$1"
  echo "=== Phase731 ${model} start $(date '+%F %T') ===" | tee "$OUT_DIR/${model}.log"
  python tests/gpt5/phase731_full_path_functional_atlas_v0.py \
    --model "$model" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee -a "$OUT_DIR/${model}.log"
  echo "=== Phase731 ${model} done $(date '+%F %T') ===" | tee -a "$OUT_DIR/${model}.log"
}

run_one qwen3
run_one glm4
run_one deepseek7b

python tests/gpt5/phase731_full_path_functional_atlas_v0.py --summarize-only \
  2>&1 | tee "$OUT_DIR/summary.log"
