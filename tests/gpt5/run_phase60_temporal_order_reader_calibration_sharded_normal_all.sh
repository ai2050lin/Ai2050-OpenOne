#!/usr/bin/env bash
set -euo pipefail

cd /home/rankrank/Documents/OpenOne/Ai2050-OpenOne

export OPENONE_NORMAL_ENV="${OPENONE_NORMAL_ENV:-openone-cu130-py312}"
export PHASE60_OUTPUT_DIR="${PHASE60_OUTPUT_DIR:-results/gpt5_phase60_temporal_order_reader_calibration_sharded_full}"
export PHASE60_MAX_CASES="${PHASE60_MAX_CASES:-384}"
export PHASE60_SHARD_CASES="${PHASE60_SHARD_CASES:-16}"
export PYTHONUNBUFFERED=1

if [[ "${CONDA_DEFAULT_ENV:-}" != "$OPENONE_NORMAL_ENV" ]]; then
  if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
  elif [[ -f /home/rankrank/miniconda3/etc/profile.d/conda.sh ]]; then
    source /home/rankrank/miniconda3/etc/profile.d/conda.sh
  else
    echo "conda was not found; cannot activate ${OPENONE_NORMAL_ENV}" >&2
    exit 2
  fi
  conda activate "$OPENONE_NORMAL_ENV"
fi

mkdir -p "$PHASE60_OUTPUT_DIR"

echo "=== Phase60 temporal order reader calibration SHARDED normal all-model ==="
date '+%Y-%m-%d %H:%M:%S %Z'
echo "conda_env=${CONDA_DEFAULT_ENV:-none}"
echo "output_dir=${PHASE60_OUTPUT_DIR}"
echo "max_cases=${PHASE60_MAX_CASES}"
echo "shard_cases=${PHASE60_SHARD_CASES}"

run_model_shards() {
  local model="$1"
  local attn_impls="${2:-flash_attention_2,sdpa,eager}"
  local offset=0
  local shard=0
  while [[ "$offset" -lt "$PHASE60_MAX_CASES" ]]; do
    local suffix
    suffix=$(printf 'shard%04d' "$shard")
    local out_file="${PHASE60_OUTPUT_DIR}/${model}_phase60_temporal_order_reader_calibration_${suffix}.json"
    if [[ -f "$out_file" ]]; then
      echo "=== Skip existing ${model} ${suffix} offset=${offset} ==="
    else
      echo
      echo "=== Run ${model}: Phase60 ${suffix}, offset=${offset}, count=${PHASE60_SHARD_CASES} ==="
      date '+%Y-%m-%d %H:%M:%S %Z'
      PHASE60_ATTN_IMPLEMENTATIONS="$attn_impls" \
        python tests/gpt5/phase60_temporal_order_reader_calibration.py "$model" \
          --output-dir "$PHASE60_OUTPUT_DIR" \
          --max-cases "$PHASE60_MAX_CASES" \
          --case-offset "$offset" \
          --case-count "$PHASE60_SHARD_CASES" \
          --output-suffix "$suffix" \
          --progress-every "${PHASE60_PROGRESS_EVERY:-8}" \
          --hard-exit-after-model
      echo "=== Completed ${model} ${suffix}; process hard-exited ==="
      sleep "${SLEEP_AFTER_SHARD:-3}"
    fi
    offset=$((offset + PHASE60_SHARD_CASES))
    shard=$((shard + 1))
  done
}

run_model_shards qwen3 "${QWEN3_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
run_model_shards glm4 "${GLM4_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
run_model_shards deepseek7b "${DEEPSEEK7B_ATTN_IMPLEMENTATIONS:-eager}"

python tests/gpt5/phase60_temporal_order_reader_calibration_summary.py \
  --input-dir "$PHASE60_OUTPUT_DIR" \
  --output-dir "$PHASE60_OUTPUT_DIR"

echo
echo "=== Phase60 sharded done ==="
date '+%Y-%m-%d %H:%M:%S %Z'
