#!/usr/bin/env bash
set -euo pipefail

cd /home/rankrank/Documents/OpenOne/Ai2050-OpenOne

export OUTPUT_DIR="${OUTPUT_DIR:-results/gpt5_phase301_passive_factor_closure}"
export MAX_BASES="${MAX_BASES:-24}"
export COMMON_ALPHAS="${COMMON_ALPHAS:-0,0.5,1.0}"
export COMMON_MODULES="${COMMON_MODULES:-resid_in,resid_out,mlp_out}"
export COMMON_PROGRESS_EVERY="${COMMON_PROGRESS_EVERY:-4}"
export COMMON_TRAIN_FRACTION="${COMMON_TRAIN_FRACTION:-0.5}"
export COMMON_MAX_SEQ_LEN="${COMMON_MAX_SEQ_LEN:-64}"
export CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

mkdir -p "$OUTPUT_DIR"

run_one() {
  local model="$1"
  local layers="$2"
  local seconds="$3"
  echo
  echo "=== Phase301 passive factor normal: ${model} ==="
  date '+%Y-%m-%d %H:%M:%S %Z'
  echo "layers=${layers}"
  echo "output_dir=${OUTPUT_DIR}"
  echo
  set +e
  MAX_SECONDS="$seconds" OUTPUT_DIR="$OUTPUT_DIR" \
    tests/gpt5/run_phase301_normal.sh "$model" \
      --max-bases "$MAX_BASES" \
      --train-fraction "$COMMON_TRAIN_FRACTION" \
      --layers "$layers" \
      --modules "$COMMON_MODULES" \
      --alphas "$COMMON_ALPHAS" \
      --max-seq-len "$COMMON_MAX_SEQ_LEN" \
      --progress-every "$COMMON_PROGRESS_EVERY"
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

run_one qwen3 "${QWEN3_LAYERS:-0,1,2,3,4,5,6,7,8}" "${QWEN3_MAX_SECONDS:-7200}"
run_one glm4 "${GLM4_LAYERS:-0,1,2,3,4,5,6,7,8}" "${GLM4_MAX_SECONDS:-10800}"
run_one deepseek7b "${DEEPSEEK7B_LAYERS:-20,21,22,23,24,25,26,27}" "${DEEPSEEK7B_MAX_SECONDS:-7200}"

echo
echo "=== Phase301 passive factor normal all done ==="
date '+%Y-%m-%d %H:%M:%S %Z'
find "$OUTPUT_DIR" -maxdepth 3 -type f | sort
