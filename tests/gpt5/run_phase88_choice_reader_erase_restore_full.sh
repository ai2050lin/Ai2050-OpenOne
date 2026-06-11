#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${PHASE88_OUTPUT_DIR:-results/gpt5_phase88_choice_reader_erase_restore_full_${TS}}"
mkdir -p "$OUT_DIR"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PHASE88_ATTN_IMPLEMENTATIONS="${PHASE88_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
export PHASE68_MAX_GPU_MEMORY="${PHASE68_MAX_GPU_MEMORY:-22GiB}"
export PHASE68_MAX_CPU_MEMORY="${PHASE68_MAX_CPU_MEMORY:-96GiB}"

MAX_ITEMS="${PHASE88_MAX_ITEMS:-336}"
MAX_BASIS_ITEMS="${PHASE88_MAX_BASIS_ITEMS:-224}"
PROGRESS_EVERY="${PHASE88_PROGRESS_EVERY:-42}"
CONDITIONS="${PHASE88_CONDITIONS:-frame_suffix_final,frame_suffix_all,frame_suffix_function,frame_suffix_lexical,frame_all_suffix_tokens,object_suffix_final,object_all_suffix_tokens}"
ORDERS="${PHASE88_CHOICE_ORDERS:-rotating,target_last}"

run_one() {
  local model="$1"
  local pairs="$2"
  local templates="$3"
  echo "[$(date '+%F %T')] Phase88 start model=${model} pairs=${pairs} templates=${templates}" | tee -a "$OUT_DIR/run.log"
  python tests/gpt5/phase88_choice_reader_erase_restore.py "$model" \
    --layer-pairs "$pairs" \
    --max-items "$MAX_ITEMS" \
    --max-basis-items "$MAX_BASIS_ITEMS" \
    --choice-templates "$templates" \
    --choice-orders "$ORDERS" \
    --conditions "$CONDITIONS" \
    --output-dir "$OUT_DIR" \
    --progress-every "$PROGRESS_EVERY" \
    --hard-exit-after-model 2>&1 | tee -a "$OUT_DIR/${model}.log"
  echo "[$(date '+%F %T')] Phase88 done model=${model}" | tee -a "$OUT_DIR/run.log"
  sleep 8
}

run_one qwen3 "4-8,8-12" "${PHASE88_QWEN3_TEMPLATES:-choice_json_letter,choice_no_explain}"
run_one glm4 "4-10,10-20" "${PHASE88_GLM4_TEMPLATES:-choice_json_letter,choice_no_explain,choice_blank}"
run_one deepseek7b "8-10,12-14" "${PHASE88_DS7B_TEMPLATES:-choice_json_letter}"

python tests/gpt5/phase88_choice_reader_erase_restore_summary.py --output-dir "$OUT_DIR" 2>&1 | tee -a "$OUT_DIR/summary.log"
echo "$OUT_DIR"
