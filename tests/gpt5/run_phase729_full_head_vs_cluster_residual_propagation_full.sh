#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="results/glm5_phase729_full_head_vs_cluster_residual_propagation"
mkdir -p "$OUT_DIR"

COMMON_ARGS=(
  --downstream-layers "${PHASE729_DOWNSTREAM_LAYERS:-8}"
  --component-layers "${PHASE729_COMPONENT_LAYERS:-4}"
  --log-every "${PHASE729_LOG_EVERY:-8}"
  --hard-exit-after-model
)

if [[ -n "${PHASE729_MAX_CASES:-}" ]]; then
  COMMON_ARGS+=(--max-cases "$PHASE729_MAX_CASES")
fi

run_one() {
  local model="$1"
  echo "=== Phase729 ${model} start $(date '+%F %T') ===" | tee "$OUT_DIR/${model}.log"
  python tests/gpt5/phase729_full_head_vs_cluster_residual_propagation.py \
    --model "$model" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee -a "$OUT_DIR/${model}.log"
  echo "=== Phase729 ${model} done $(date '+%F %T') ===" | tee -a "$OUT_DIR/${model}.log"
}

run_one qwen3
run_one glm4
run_one deepseek7b

python tests/gpt5/phase729_full_head_vs_cluster_residual_propagation.py --summarize-only \
  2>&1 | tee "$OUT_DIR/summary.log"
