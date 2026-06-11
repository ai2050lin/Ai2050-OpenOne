#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${PHASE90_OUTPUT_DIR:-results/gpt5_phase90_component_margin_reader_alignment_full_${TS}}"
mkdir -p "$OUT_DIR"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PHASE90_ATTN_IMPLEMENTATIONS="${PHASE90_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
export PHASE68_MAX_GPU_MEMORY="${PHASE68_MAX_GPU_MEMORY:-22GiB}"
export PHASE68_MAX_CPU_MEMORY="${PHASE68_MAX_CPU_MEMORY:-96GiB}"

MAX_ITEMS="${PHASE90_MAX_ITEMS:-420}"
PROGRESS_EVERY="${PHASE90_PROGRESS_EVERY:-70}"
SLOTS="${PHASE90_SLOTS:-category,color,function,material,location}"
GENERATE_FLAG="${PHASE90_GENERATE_FLAG:---generate}"

run_one() {
  local model="$1"
  local layers="$2"
  echo "[$(date '+%F %T')] Phase90 start model=${model} layers=${layers}" | tee -a "$OUT_DIR/run.log"
  python tests/gpt5/phase90_component_margin_reader_alignment.py "$model" \
    --layers "$layers" \
    --slots "$SLOTS" \
    --max-items "$MAX_ITEMS" \
    --choice-template "${PHASE90_CHOICE_TEMPLATE:-choice_json_letter}" \
    --output-dir "$OUT_DIR" \
    --progress-every "$PROGRESS_EVERY" \
    $GENERATE_FLAG \
    --hard-exit-after-model 2>&1 | tee -a "$OUT_DIR/${model}.log"
  echo "[$(date '+%F %T')] Phase90 done model=${model}" | tee -a "$OUT_DIR/run.log"
  sleep 8
}

run_one qwen3 "${PHASE90_QWEN3_LAYERS:-6,12,24,27,30,35}"
run_one glm4 "${PHASE90_GLM4_LAYERS:-6,18,30,36,38,39}"
run_one deepseek7b "${PHASE90_DS7B_LAYERS:-4,8,14,24,26,27}"

python tests/gpt5/phase90_component_margin_reader_alignment_summary.py --output-dir "$OUT_DIR" 2>&1 | tee -a "$OUT_DIR/summary.log"
echo "$OUT_DIR"
