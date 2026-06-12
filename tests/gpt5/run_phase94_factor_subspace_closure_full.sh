#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${PHASE94_OUTPUT_DIR:-results/gpt5_phase94_factor_subspace_closure_full_${TS}}"
mkdir -p "$OUT_DIR"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PHASE94_ATTN_IMPLEMENTATIONS="${PHASE94_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
export PHASE68_MAX_GPU_MEMORY="${PHASE68_MAX_GPU_MEMORY:-22GiB}"
export PHASE68_MAX_CPU_MEMORY="${PHASE68_MAX_CPU_MEMORY:-96GiB}"

MAX_ITEMS="${PHASE94_MAX_ITEMS:-420}"
PROGRESS_EVERY="${PHASE94_PROGRESS_EVERY:-70}"
SLOTS="${PHASE94_SLOTS:-category,color,function,material,location}"
RANK="${PHASE94_RANK:-4}"
POOL_MODE="${PHASE94_POOL_MODE:-tail}"
COPY_MODE="${PHASE94_COPY_MODE:-both}"

run_one() {
  local model="$1"
  local nodes="$2"
  echo "[$(date '+%F %T')] Phase94 start model=${model} nodes=${nodes}" | tee -a "$OUT_DIR/run.log"
  python tests/gpt5/phase94_factor_subspace_closure.py "$model" \
    --nodes "$nodes" \
    --slots "$SLOTS" \
    --max-items "$MAX_ITEMS" \
    --rank "$RANK" \
    --pool-mode "$POOL_MODE" \
    --copy-mode "$COPY_MODE" \
    --choice-template "${PHASE94_CHOICE_TEMPLATE:-choice_json_letter}" \
    --output-dir "$OUT_DIR" \
    --progress-every "$PROGRESS_EVERY" \
    --hard-exit-after-model 2>&1 | tee -a "$OUT_DIR/${model}.log"
  echo "[$(date '+%F %T')] Phase94 done model=${model}" | tee -a "$OUT_DIR/run.log"
  sleep 8
}

run_one qwen3 "${PHASE94_QWEN3_NODES:-6:mlp,24:attn}"
run_one glm4 "${PHASE94_GLM4_NODES:-39:mlp}"
run_one deepseek7b "${PHASE94_DS7B_NODES:-26:mlp,27:attn}"

python tests/gpt5/phase94_factor_subspace_closure_summary.py --output-dir "$OUT_DIR" 2>&1 | tee -a "$OUT_DIR/summary.log"
echo "$OUT_DIR"
