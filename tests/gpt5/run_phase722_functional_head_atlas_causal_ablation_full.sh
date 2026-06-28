#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT_DIR="results/glm5_phase722_functional_head_atlas_causal_ablation"
mkdir -p "$OUT_DIR"

run_one() {
  local model="$1"
  echo "=== Phase 722 ${model} $(date '+%Y-%m-%d %H:%M:%S') ===" | tee -a "$OUT_DIR/run_phase722.log"
  python tests/gpt5/phase722_functional_head_atlas_causal_ablation.py \
    --model "$model" \
    --top-heads-per-family "${PHASE722_TOP_HEADS_PER_FAMILY:-3}" \
    --max-cases-per-family "${PHASE722_MAX_CASES_PER_FAMILY:-24}" \
    --log-every "${PHASE722_LOG_EVERY:-8}" \
    --hard-exit-after-model 2>&1 | tee -a "$OUT_DIR/${model}.log"
}

run_one qwen3
run_one glm4
run_one deepseek7b

python tests/gpt5/phase722_functional_head_atlas_causal_ablation.py --summarize-only \
  2>&1 | tee -a "$OUT_DIR/summary.log"
