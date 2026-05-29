#!/usr/bin/env bash
set -euo pipefail

cd /home/rankrank/Documents/OpenOne/Ai2050-OpenOne

export OUTPUT_DIR="${OUTPUT_DIR:-results/gpt5_phase298_expanded_dynamic_normal}"
export COMMON_CATEGORIES="${COMMON_CATEGORIES:-negation,logical,passive,recursive,translation,tense,coreference,style}"
export MAX_PAIRS_PER_SUBTYPE="${MAX_PAIRS_PER_SUBTYPE:-2}"
export COMMON_ALPHAS="${COMMON_ALPHAS:-0,0.25,0.5,0.75,1.0}"
export COMMON_LABEL="${COMMON_LABEL:-expanded_dynamic_normal}"
export COMMON_PROGRESS_EVERY="${COMMON_PROGRESS_EVERY:-8}"
export COMMON_MAX_SEQ_LEN="${COMMON_MAX_SEQ_LEN:-64}"
export COMMON_PATCH_TYPES="${COMMON_PATCH_TYPES:-resid_in,resid_out,attn_out,mlp_out}"
export CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

mkdir -p "$OUTPUT_DIR"

run_one() {
  local model="$1"
  local layers="$2"
  local seconds="$3"
  echo
  echo "=== GSSC expanded dynamic normal: ${model} ==="
  date '+%Y-%m-%d %H:%M:%S %Z'
  echo "layers=${layers}"
  echo "output_dir=${OUTPUT_DIR}"
  echo
  set +e
  MAX_SECONDS="$seconds" OUTPUT_DIR="$OUTPUT_DIR" \
    tests/gpt5/run_phase294_normal.sh "$model" \
      --categories "$COMMON_CATEGORIES" \
      --max-pairs-per-subtype "$MAX_PAIRS_PER_SUBTYPE" \
      --layers "$layers" \
      --alphas "$COMMON_ALPHAS" \
      --patch-types "$COMMON_PATCH_TYPES" \
      --max-seq-len "$COMMON_MAX_SEQ_LEN" \
      --progress-every "$COMMON_PROGRESS_EVERY" \
      --label "$COMMON_LABEL"
  local rc="$?"
  set -e
  echo "model=${model} rc=${rc}"
  echo "--- post-model compute apps ---"
  timeout 8s nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits || true
  sleep 8
  if [[ "$rc" != "0" && "$CONTINUE_ON_ERROR" != "1" ]]; then
    exit "$rc"
  fi
}

run_one qwen3 "${QWEN3_LAYERS:-0,1,2,3,4,5,6,7,8}" "${QWEN3_MAX_SECONDS:-9000}"
run_one glm4 "${GLM4_LAYERS:-0,1,2,3,4,5,6,7,8}" "${GLM4_MAX_SECONDS:-10800}"
run_one deepseek7b "${DEEPSEEK7B_LAYERS:-20,21,22,23,24,25,26,27}" "${DEEPSEEK7B_MAX_SECONDS:-9000}"

echo
echo "=== GSSC expanded dynamic normal all done ==="
date '+%Y-%m-%d %H:%M:%S %Z'
find "$OUTPUT_DIR" -maxdepth 4 -type f | sort
