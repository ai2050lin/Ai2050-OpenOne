#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${PHASE93_OUTPUT_DIR:-results/gpt5_phase93_transplant_alignment_audit_full_${TS}}"
mkdir -p "$OUT_DIR"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PHASE93_ATTN_IMPLEMENTATIONS="${PHASE93_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
export PHASE68_MAX_GPU_MEMORY="${PHASE68_MAX_GPU_MEMORY:-22GiB}"
export PHASE68_MAX_CPU_MEMORY="${PHASE68_MAX_CPU_MEMORY:-96GiB}"

MAX_ITEMS="${PHASE93_MAX_ITEMS:-420}"
PROGRESS_EVERY="${PHASE93_PROGRESS_EVERY:-70}"
SLOTS="${PHASE93_SLOTS:-category,color,function,material,location}"
COPY_MODES="${PHASE93_COPY_MODES:-tail,prefix,both}"
DONOR_KINDS="${PHASE93_DONOR_KINDS:-self_restore,same_slot_same_target,same_slot_diff_target,diff_slot_same_object,diff_slot_diff_object}"

run_one() {
  local model="$1"
  local nodes="$2"
  echo "[$(date '+%F %T')] Phase93 start model=${model} nodes=${nodes}" | tee -a "$OUT_DIR/run.log"
  python tests/gpt5/phase93_transplant_alignment_audit.py "$model" \
    --nodes "$nodes" \
    --slots "$SLOTS" \
    --max-items "$MAX_ITEMS" \
    --choice-template "${PHASE93_CHOICE_TEMPLATE:-choice_json_letter}" \
    --copy-modes "$COPY_MODES" \
    --donor-kinds "$DONOR_KINDS" \
    --output-dir "$OUT_DIR" \
    --progress-every "$PROGRESS_EVERY" \
    --hard-exit-after-model 2>&1 | tee -a "$OUT_DIR/${model}.log"
  echo "[$(date '+%F %T')] Phase93 done model=${model}" | tee -a "$OUT_DIR/run.log"
  sleep 8
}

run_one qwen3 "${PHASE93_QWEN3_NODES:-6:mlp,24:attn}"
run_one glm4 "${PHASE93_GLM4_NODES:-39:mlp}"
run_one deepseek7b "${PHASE93_DS7B_NODES:-26:mlp,27:attn}"

python tests/gpt5/phase93_transplant_alignment_audit_summary.py --output-dir "$OUT_DIR" 2>&1 | tee -a "$OUT_DIR/summary.log"
echo "$OUT_DIR"
