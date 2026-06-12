#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${PHASE98_OUTPUT_DIR:-results/gpt5_phase98_head_readout_route_mapping_full_${TS}}"
mkdir -p "$OUT_DIR"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PHASE98_ATTN_IMPLEMENTATIONS="${PHASE98_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
export PHASE68_MAX_GPU_MEMORY="${PHASE68_MAX_GPU_MEMORY:-22GiB}"
export PHASE68_MAX_CPU_MEMORY="${PHASE68_MAX_CPU_MEMORY:-96GiB}"

MAX_ITEMS="${PHASE98_MAX_ITEMS:-105}"
PROGRESS_EVERY="${PHASE98_PROGRESS_EVERY:-35}"
SLOTS="${PHASE98_SLOTS:-category,color,function,material,location}"
POSITIONS="${PHASE98_POSITIONS:-prompt_tail,last4}"
DONOR_KINDS="${PHASE98_DONOR_KINDS:-same_slot_diff_target}"
HEADS="${PHASE98_HEADS:-all}"

run_one() {
  local model="$1"
  local layers="$2"
  echo "[$(date '+%F %T')] Phase98 start model=${model} layers=${layers}" | tee -a "$OUT_DIR/run.log"
  python tests/gpt5/phase98_head_readout_route_mapping.py "$model" \
    --layers "$layers" \
    --heads "$HEADS" \
    --slots "$SLOTS" \
    --max-items "$MAX_ITEMS" \
    --positions "$POSITIONS" \
    --donor-kinds "$DONOR_KINDS" \
    --choice-template "${PHASE98_CHOICE_TEMPLATE:-choice_json_letter}" \
    --output-dir "$OUT_DIR" \
    --progress-every "$PROGRESS_EVERY" \
    --hard-exit-after-model 2>&1 | tee -a "$OUT_DIR/${model}.log"
  echo "[$(date '+%F %T')] Phase98 done model=${model}" | tee -a "$OUT_DIR/run.log"
  sleep 8
}

run_one qwen3 "${PHASE98_QWEN3_LAYERS:-24}"
run_one glm4 "${PHASE98_GLM4_LAYERS:-39}"
run_one deepseek7b "${PHASE98_DS7B_LAYERS:-27}"

python tests/gpt5/phase98_head_readout_route_mapping_summary.py --output-dir "$OUT_DIR" 2>&1 | tee -a "$OUT_DIR/summary.log"
echo "$OUT_DIR"
