#!/usr/bin/env bash
set -euo pipefail

cd /home/rankrank/Documents/OpenOne/Ai2050-OpenOne

export OPENONE_NORMAL_ENV="${OPENONE_NORMAL_ENV:-openone-cu130-py312}"
export PHASE56_OUTPUT_DIR="${PHASE56_OUTPUT_DIR:-results/gpt5_phase56_global_path_interaction_full}"
export PHASE344_OUTPUT_DIR="${PHASE56_OUTPUT_DIR}/phase344_345"
export PHASE346_OUTPUT_DIR="${PHASE56_OUTPUT_DIR}/phase346"
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

mkdir -p "$PHASE344_OUTPUT_DIR" "$PHASE346_OUTPUT_DIR"

echo "=== Phase56 global path + MLP interaction normal all-model ==="
date '+%Y-%m-%d %H:%M:%S %Z'
echo "conda_env=${CONDA_DEFAULT_ENV:-none}"
echo "output_dir=${PHASE56_OUTPUT_DIR}"
echo "nvidia_driver=$(cat /proc/driver/nvidia/version 2>/dev/null | head -n 1 || true)"

run_model() {
  local model="$1"
  local attn_impls="${2:-flash_attention_2,sdpa,eager}"
  echo
  echo "=== Run ${model}: Phase344+345 multi-relation ==="
  date '+%Y-%m-%d %H:%M:%S %Z'
  PHASE344_ATTN_IMPLEMENTATIONS="$attn_impls" \
    python tests/glm5/phase344_345_multi_relation.py "$model" \
      --output-dir "$PHASE344_OUTPUT_DIR" \
      --hard-exit-after-model

  sleep "${SLEEP_AFTER_MODEL:-5}"

  echo
  echo "=== Run ${model}: Phase346 interaction closure ==="
  date '+%Y-%m-%d %H:%M:%S %Z'
  PHASE346_ATTN_IMPLEMENTATIONS="$attn_impls" \
    python tests/glm5/phase346_interaction_closure.py "$model" \
      --output-dir "$PHASE346_OUTPUT_DIR" \
      --hard-exit-after-model

  echo "=== Completed ${model}; process hard-exited ==="
  sleep "${SLEEP_AFTER_MODEL:-5}"
}

run_model qwen3 "${QWEN3_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
run_model glm4 "${GLM4_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
run_model deepseek7b "${DEEPSEEK7B_ATTN_IMPLEMENTATIONS:-eager}"

python tests/gpt5/phase56_global_path_interaction_summary.py \
  --phase344-dir "$PHASE344_OUTPUT_DIR" \
  --phase346-dir "$PHASE346_OUTPUT_DIR" \
  --output-dir "$PHASE56_OUTPUT_DIR"

echo
echo "=== Phase56 done ==="
date '+%Y-%m-%d %H:%M:%S %Z'
