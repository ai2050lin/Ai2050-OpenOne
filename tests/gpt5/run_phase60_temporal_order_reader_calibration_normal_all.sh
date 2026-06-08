#!/usr/bin/env bash
set -euo pipefail

cd /home/rankrank/Documents/OpenOne/Ai2050-OpenOne

export OPENONE_NORMAL_ENV="${OPENONE_NORMAL_ENV:-openone-cu130-py312}"
export PHASE60_OUTPUT_DIR="${PHASE60_OUTPUT_DIR:-results/gpt5_phase60_temporal_order_reader_calibration_full}"
export PHASE60_MAX_CASES="${PHASE60_MAX_CASES:-384}"
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

echo "=== Phase60 temporal order reader calibration normal all-model ==="
date '+%Y-%m-%d %H:%M:%S %Z'
echo "conda_env=${CONDA_DEFAULT_ENV:-none}"
echo "output_dir=${PHASE60_OUTPUT_DIR}"
echo "max_cases=${PHASE60_MAX_CASES}"

run_model() {
  local model="$1"
  local attn_impls="${2:-flash_attention_2,sdpa,eager}"
  echo
  echo "=== Run ${model}: Phase60 temporal order reader calibration ==="
  date '+%Y-%m-%d %H:%M:%S %Z'
  PHASE60_ATTN_IMPLEMENTATIONS="$attn_impls" \
    python tests/gpt5/phase60_temporal_order_reader_calibration.py "$model" \
      --output-dir "$PHASE60_OUTPUT_DIR" \
      --max-cases "$PHASE60_MAX_CASES" \
      --progress-every "${PHASE60_PROGRESS_EVERY:-24}" \
      --hard-exit-after-model
  echo "=== Completed ${model}; process hard-exited ==="
  sleep "${SLEEP_AFTER_MODEL:-5}"
}

run_model qwen3 "${QWEN3_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
run_model glm4 "${GLM4_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
run_model deepseek7b "${DEEPSEEK7B_ATTN_IMPLEMENTATIONS:-eager}"

python tests/gpt5/phase60_temporal_order_reader_calibration_summary.py \
  --input-dir "$PHASE60_OUTPUT_DIR" \
  --output-dir "$PHASE60_OUTPUT_DIR"

echo
echo "=== Phase60 done ==="
date '+%Y-%m-%d %H:%M:%S %Z'
