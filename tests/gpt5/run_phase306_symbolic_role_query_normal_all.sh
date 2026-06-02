#!/usr/bin/env bash
set -euo pipefail

cd /home/rankrank/Documents/OpenOne/Ai2050-OpenOne

export OUTPUT_DIR="${OUTPUT_DIR:-results/gpt5_phase306_symbolic_role_query_calibration}"
export ENABLE_GPU_MONITOR="${ENABLE_GPU_MONITOR:-0}"
export ENABLE_SNAPSHOT_NVIDIA_SMI="${ENABLE_SNAPSHOT_NVIDIA_SMI:-1}"
export PROBE_ATTN_IMPLEMENTATION="${PROBE_ATTN_IMPLEMENTATION:-sdpa}"
export PROBE_DEVICE_MAP_AUTO_MODELS="${PROBE_DEVICE_MAP_AUTO_MODELS:-glm4,deepseek7b}"
export PROBE_MAX_GPU_MEMORY="${PROBE_MAX_GPU_MEMORY:-21GiB}"
export PROBE_MAX_CPU_MEMORY="${PROBE_MAX_CPU_MEMORY:-96GiB}"

COMMON_ARGS=(
  --max-bases "${MAX_BASES:-32}"
  --max-seq-len "${MAX_SEQ_LEN:-128}"
  --entity-styles "${ENTITY_STYLES:-ab,entity_ab,nonce}"
  --answer-styles "${ANSWER_STYLES:-letter,entity}"
  --progress-every "${PROGRESS_EVERY:-4}"
)

echo "=== Phase306 all-model symbolic role query calibration ==="
date '+%Y-%m-%d %H:%M:%S %Z'
echo "output_dir=${OUTPUT_DIR}"
echo "common_args=${COMMON_ARGS[*]}"

run_model() {
  local model="$1"
  local max_seconds="$2"
  echo
  echo "=== Run ${model} ==="
  MAX_SECONDS="$max_seconds" tests/gpt5/run_phase306_normal.sh "$model" "${COMMON_ARGS[@]}"
  echo "=== Completed ${model}; sleep for unload ==="
  sleep "${SLEEP_AFTER_MODEL:-10}"
}

run_model qwen3 "${QWEN3_MAX_SECONDS:-3600}"
run_model glm4 "${GLM4_MAX_SECONDS:-4200}"
run_model deepseek7b "${DEEPSEEK7B_MAX_SECONDS:-4200}"

python tests/gpt5/phase306_symbolic_role_query_summary.py \
  --input-dir "$OUTPUT_DIR" \
  --output-dir "$OUTPUT_DIR"

echo "=== Phase306 all-model done ==="
date '+%Y-%m-%d %H:%M:%S %Z'
