#!/usr/bin/env bash
set -euo pipefail

cd /home/rankrank/Documents/OpenOne/Ai2050-OpenOne

export OPENONE_NORMAL_ENV="${OPENONE_NORMAL_ENV:-openone-cu130-py312}"
export PHASE57_OUTPUT_DIR="${PHASE57_OUTPUT_DIR:-results/gpt5_phase57_global_relation_path_matrix_full}"
export PHASE57_MAX_PAIRS_PER_RELATION="${PHASE57_MAX_PAIRS_PER_RELATION:-40}"
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

mkdir -p "$PHASE57_OUTPUT_DIR"

echo "=== Phase57 global relation path matrix normal all-model ==="
date '+%Y-%m-%d %H:%M:%S %Z'
echo "conda_env=${CONDA_DEFAULT_ENV:-none}"
echo "output_dir=${PHASE57_OUTPUT_DIR}"
echo "max_pairs_per_relation=${PHASE57_MAX_PAIRS_PER_RELATION}"
echo "nvidia_driver=$(cat /proc/driver/nvidia/version 2>/dev/null | head -n 1 || true)"

run_model() {
  local model="$1"
  local attn_impls="${2:-flash_attention_2,sdpa,eager}"
  echo
  echo "=== Run ${model}: Phase57 global relation path matrix ==="
  date '+%Y-%m-%d %H:%M:%S %Z'
  PHASE57_ATTN_IMPLEMENTATIONS="$attn_impls" \
    python tests/gpt5/phase57_global_relation_path_matrix.py "$model" \
      --output-dir "$PHASE57_OUTPUT_DIR" \
      --max-pairs-per-relation "$PHASE57_MAX_PAIRS_PER_RELATION" \
      --hard-exit-after-model
  echo "=== Completed ${model}; process hard-exited ==="
  sleep "${SLEEP_AFTER_MODEL:-5}"
}

run_model qwen3 "${QWEN3_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
run_model glm4 "${GLM4_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
run_model deepseek7b "${DEEPSEEK7B_ATTN_IMPLEMENTATIONS:-eager}"

python tests/gpt5/phase57_global_relation_path_matrix_summary.py \
  --input-dir "$PHASE57_OUTPUT_DIR" \
  --output-dir "$PHASE57_OUTPUT_DIR"

echo
echo "=== Phase57 done ==="
date '+%Y-%m-%d %H:%M:%S %Z'
