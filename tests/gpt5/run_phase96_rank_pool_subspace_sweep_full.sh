#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${PHASE96_OUTPUT_DIR:-results/gpt5_phase96_rank_pool_subspace_sweep_full_${TS}}"
mkdir -p "$OUT_DIR"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PHASE96_ATTN_IMPLEMENTATIONS="${PHASE96_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
export PHASE68_MAX_GPU_MEMORY="${PHASE68_MAX_GPU_MEMORY:-22GiB}"
export PHASE68_MAX_CPU_MEMORY="${PHASE68_MAX_CPU_MEMORY:-96GiB}"

MAX_ITEMS="${PHASE96_MAX_ITEMS:-210}"
PROGRESS_EVERY="${PHASE96_PROGRESS_EVERY:-35}"
SLOTS="${PHASE96_SLOTS:-category,color,function,material,location}"
RANKS="${PHASE96_RANKS:-1,4,16}"
POOL_MODES="${PHASE96_POOL_MODES:-tail,prefix,mean}"
FACTORS="${PHASE96_FACTORS:-pc1,object,target,slot,choice}"

run_one() {
  local model="$1"
  local nodes="$2"
  echo "[$(date '+%F %T')] Phase96 start model=${model} nodes=${nodes}" | tee -a "$OUT_DIR/run.log"
  python tests/gpt5/phase96_rank_pool_subspace_sweep.py "$model" \
    --nodes "$nodes" \
    --slots "$SLOTS" \
    --max-items "$MAX_ITEMS" \
    --ranks "$RANKS" \
    --pool-modes "$POOL_MODES" \
    --factors "$FACTORS" \
    --choice-template "${PHASE96_CHOICE_TEMPLATE:-choice_json_letter}" \
    --output-dir "$OUT_DIR" \
    --progress-every "$PROGRESS_EVERY" \
    --hard-exit-after-model 2>&1 | tee -a "$OUT_DIR/${model}.log"
  echo "[$(date '+%F %T')] Phase96 done model=${model}" | tee -a "$OUT_DIR/run.log"
  sleep 8
}

run_one qwen3 "${PHASE96_QWEN3_NODES:-6:mlp,24:attn}"
run_one glm4 "${PHASE96_GLM4_NODES:-39:mlp}"
run_one deepseek7b "${PHASE96_DS7B_NODES:-27:attn}"

python tests/gpt5/phase96_rank_pool_subspace_sweep_summary.py --output-dir "$OUT_DIR" 2>&1 | tee -a "$OUT_DIR/summary.log"
echo "$OUT_DIR"
