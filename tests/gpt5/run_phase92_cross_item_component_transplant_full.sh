#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${PHASE92_OUTPUT_DIR:-results/gpt5_phase92_cross_item_component_transplant_full_${TS}}"
mkdir -p "$OUT_DIR"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PHASE92_ATTN_IMPLEMENTATIONS="${PHASE92_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
export PHASE68_MAX_GPU_MEMORY="${PHASE68_MAX_GPU_MEMORY:-22GiB}"
export PHASE68_MAX_CPU_MEMORY="${PHASE68_MAX_CPU_MEMORY:-96GiB}"

MAX_ITEMS="${PHASE92_MAX_ITEMS:-420}"
PROGRESS_EVERY="${PHASE92_PROGRESS_EVERY:-70}"
SLOTS="${PHASE92_SLOTS:-category,color,function,material,location}"
COPY_MODE="${PHASE92_COPY_MODE:-tail}"

run_one() {
  local model="$1"
  local nodes="$2"
  echo "[$(date '+%F %T')] Phase92 start model=${model} nodes=${nodes}" | tee -a "$OUT_DIR/run.log"
  python tests/gpt5/phase92_cross_item_component_transplant.py "$model" \
    --nodes "$nodes" \
    --slots "$SLOTS" \
    --max-items "$MAX_ITEMS" \
    --choice-template "${PHASE92_CHOICE_TEMPLATE:-choice_json_letter}" \
    --copy-mode "$COPY_MODE" \
    --output-dir "$OUT_DIR" \
    --progress-every "$PROGRESS_EVERY" \
    --hard-exit-after-model 2>&1 | tee -a "$OUT_DIR/${model}.log"
  echo "[$(date '+%F %T')] Phase92 done model=${model}" | tee -a "$OUT_DIR/run.log"
  sleep 8
}

run_one qwen3 "${PHASE92_QWEN3_NODES:-6:mlp,24:attn}"
run_one glm4 "${PHASE92_GLM4_NODES:-39:mlp}"
run_one deepseek7b "${PHASE92_DS7B_NODES:-26:mlp,27:attn}"

python tests/gpt5/phase92_cross_item_component_transplant_summary.py --output-dir "$OUT_DIR" 2>&1 | tee -a "$OUT_DIR/summary.log"
echo "$OUT_DIR"
