#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${PHASE97_OUTPUT_DIR:-results/gpt5_phase97_token_route_local_patch_full_${TS}}"
mkdir -p "$OUT_DIR"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PHASE97_ATTN_IMPLEMENTATIONS="${PHASE97_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
export PHASE68_MAX_GPU_MEMORY="${PHASE68_MAX_GPU_MEMORY:-22GiB}"
export PHASE68_MAX_CPU_MEMORY="${PHASE68_MAX_CPU_MEMORY:-96GiB}"

MAX_ITEMS="${PHASE97_MAX_ITEMS:-210}"
PROGRESS_EVERY="${PHASE97_PROGRESS_EVERY:-35}"
SLOTS="${PHASE97_SLOTS:-category,color,function,material,location}"
POSITIONS="${PHASE97_POSITIONS:-object_span,relation_span,prompt_tail,last4,prefix8}"
DONOR_KINDS="${PHASE97_DONOR_KINDS:-same_slot_same_target,same_slot_diff_target}"

run_one() {
  local model="$1"
  local nodes="$2"
  echo "[$(date '+%F %T')] Phase97 start model=${model} nodes=${nodes}" | tee -a "$OUT_DIR/run.log"
  python tests/gpt5/phase97_token_route_local_patch.py "$model" \
    --nodes "$nodes" \
    --slots "$SLOTS" \
    --max-items "$MAX_ITEMS" \
    --positions "$POSITIONS" \
    --donor-kinds "$DONOR_KINDS" \
    --choice-template "${PHASE97_CHOICE_TEMPLATE:-choice_json_letter}" \
    --output-dir "$OUT_DIR" \
    --progress-every "$PROGRESS_EVERY" \
    --hard-exit-after-model 2>&1 | tee -a "$OUT_DIR/${model}.log"
  echo "[$(date '+%F %T')] Phase97 done model=${model}" | tee -a "$OUT_DIR/run.log"
  sleep 8
}

run_one qwen3 "${PHASE97_QWEN3_NODES:-24:attn,6:mlp}"
run_one glm4 "${PHASE97_GLM4_NODES:-39:mlp}"
run_one deepseek7b "${PHASE97_DS7B_NODES:-27:attn}"

python tests/gpt5/phase97_token_route_local_patch_summary.py --output-dir "$OUT_DIR" 2>&1 | tee -a "$OUT_DIR/summary.log"
echo "$OUT_DIR"
