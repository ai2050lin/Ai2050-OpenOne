#!/usr/bin/env bash
set -euo pipefail

cd /home/rankrank/Documents/OpenOne/Ai2050-OpenOne

export OUTPUT_DIR="${OUTPUT_DIR:-results/gpt5_phase302_passive_token_role_sharded}"
export MAX_BASES="${MAX_BASES:-16}"
export TEST_TOTAL="${TEST_TOTAL:-8}"
export SHARD_SIZE="${SHARD_SIZE:-1}"
export COMMON_ARGS="${COMMON_ARGS:---max-bases ${MAX_BASES} --train-fraction 0.5 --modules resid_in,resid_out,mlp_out --alphas 0,1.0 --progress-every 1}"
export ENABLE_GPU_MONITOR="${ENABLE_GPU_MONITOR:-0}"
export ENABLE_SNAPSHOT_NVIDIA_SMI="${ENABLE_SNAPSHOT_NVIDIA_SMI:-1}"

echo "=== Phase302 sharded all-model normal run ==="
date '+%Y-%m-%d %H:%M:%S %Z'
echo "output_dir=${OUTPUT_DIR}"
echo "max_bases=${MAX_BASES}"
echo "test_total=${TEST_TOTAL}"
echo "shard_size=${SHARD_SIZE}"
echo "common_args=${COMMON_ARGS}"
echo "enable_gpu_monitor=${ENABLE_GPU_MONITOR}"
echo "enable_snapshot_nvidia_smi=${ENABLE_SNAPSHOT_NVIDIA_SMI}"
echo

run_model_shards() {
  local model="$1"
  local layers="$2"
  local max_seconds="$3"
  local start=0
  while [[ "$start" -lt "$TEST_TOTAL" ]]; do
    local count="$SHARD_SIZE"
    if (( start + count > TEST_TOTAL )); then
      count=$(( TEST_TOTAL - start ))
    fi
    local label
    label="$(printf 'test%03d-%03d' "$start" "$((start + count))")"
    echo "=== Run ${model} shard ${label} ==="
    MAX_SECONDS="$max_seconds" OUTPUT_DIR="$OUTPUT_DIR" \
      tests/gpt5/run_phase302_normal.sh "$model" \
        --layers "$layers" \
        --test-start "$start" \
        --test-count "$count" \
        --shard-label "$label" \
        ${COMMON_ARGS}
    start=$(( start + count ))
  done
}

run_model_shards qwen3 "${QWEN3_LAYERS:-0,1,2,3,4,5,6,7,8}" "${QWEN3_MAX_SECONDS:-1800}"
run_model_shards glm4 "${GLM4_LAYERS:-0,1,2,3,4,5,6,7,8}" "${GLM4_MAX_SECONDS:-2400}"
run_model_shards deepseek7b "${DEEPSEEK7B_LAYERS:-20,21,22,23,24,25,26,27}" "${DEEPSEEK7B_MAX_SECONDS:-1800}"

echo "=== Summarize shards ==="
python tests/gpt5/phase302_passive_token_role_summary.py \
  --input-dir "$OUTPUT_DIR" \
  --output-dir "$OUTPUT_DIR"

echo "=== Done Phase302 sharded all-model normal run ==="
date '+%Y-%m-%d %H:%M:%S %Z'
