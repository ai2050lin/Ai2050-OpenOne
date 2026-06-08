#!/usr/bin/env bash
set -euo pipefail

cd /home/rankrank/Documents/OpenOne/Ai2050-OpenOne

export OPENONE_NORMAL_ENV="${OPENONE_NORMAL_ENV:-openone-cu130-py312}"
export PHASE64_OUTPUT_DIR="${PHASE64_OUTPUT_DIR:-results/gpt5_phase64_same_class_reader_refine_sharded_full}"
export PHASE64_MAX_CASES="${PHASE64_MAX_CASES:-384}"
export PHASE64_SHARD_CASES="${PHASE64_SHARD_CASES:-16}"
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

mkdir -p "$PHASE64_OUTPUT_DIR"

echo "=== Phase64 same class reader refine SHARDED normal all-model ==="
date '+%Y-%m-%d %H:%M:%S %Z'
echo "conda_env=${CONDA_DEFAULT_ENV:-none}"
echo "output_dir=${PHASE64_OUTPUT_DIR}"
echo "max_cases=${PHASE64_MAX_CASES}"
echo "shard_cases=${PHASE64_SHARD_CASES}"

run_model_shards() {
  local model="$1"
  local attn_impls="${2:-flash_attention_2,sdpa,eager}"
  local offset=0
  local shard=0
  while [[ "$offset" -lt "$PHASE64_MAX_CASES" ]]; do
    local suffix
    suffix=$(printf 'shard%04d' "$shard")
    local out_file="${PHASE64_OUTPUT_DIR}/${model}_phase64_same_class_reader_refine_${suffix}.json"
    if [[ -f "$out_file" ]]; then
      echo "=== Skip existing ${model} ${suffix} offset=${offset} ==="
    else
      echo
      echo "=== Run ${model}: Phase64 ${suffix}, offset=${offset}, count=${PHASE64_SHARD_CASES} ==="
      date '+%Y-%m-%d %H:%M:%S %Z'
      PHASE64_ATTN_IMPLEMENTATIONS="$attn_impls" \
        python tests/gpt5/phase64_same_class_reader_refine.py "$model" \
          --output-dir "$PHASE64_OUTPUT_DIR" \
          --max-cases "$PHASE64_MAX_CASES" \
          --case-offset "$offset" \
          --case-count "$PHASE64_SHARD_CASES" \
          --output-suffix "$suffix" \
          --progress-every "${PHASE64_PROGRESS_EVERY:-8}" \
          --hard-exit-after-model
      echo "=== Completed ${model} ${suffix}; process hard-exited ==="
      sleep "${SLEEP_AFTER_SHARD:-1}"
    fi
    offset=$((offset + PHASE64_SHARD_CASES))
    shard=$((shard + 1))
  done
}

run_model_shards qwen3 "${QWEN3_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
run_model_shards glm4 "${GLM4_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
run_model_shards deepseek7b "${DEEPSEEK7B_ATTN_IMPLEMENTATIONS:-eager}"

python tests/gpt5/phase64_same_class_reader_refine_summary.py \
  --input-dir "$PHASE64_OUTPUT_DIR" \
  --output-dir "$PHASE64_OUTPUT_DIR"

echo
echo "=== Phase64 sharded done ==="
date '+%Y-%m-%d %H:%M:%S %Z'
