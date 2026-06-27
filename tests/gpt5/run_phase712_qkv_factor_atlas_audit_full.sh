#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT_DIR="results/glm5_phase712_qkv_factor_atlas_audit"
mkdir -p "$OUT_DIR"

run_one() {
  local model="$1"
  echo "=== Phase 712 ${model} $(date '+%Y-%m-%d %H:%M:%S') ===" | tee -a "$OUT_DIR/run_phase712.log"
  python tests/gpt5/phase712_qkv_factor_atlas_audit.py \
    --model "$model" \
    --top-heads "${PHASE712_TOP_HEADS:-32}" \
    --channel-count "${PHASE712_CHANNEL_COUNT:-512}" \
    --log-every "${PHASE712_LOG_EVERY:-12}" \
    --hard-exit-after-model 2>&1 | tee -a "$OUT_DIR/${model}.log"
}

run_one qwen3
run_one glm4
run_one deepseek7b

python tests/gpt5/phase712_qkv_factor_atlas_audit.py --summarize-only 2>&1 | tee -a "$OUT_DIR/summary.log"
