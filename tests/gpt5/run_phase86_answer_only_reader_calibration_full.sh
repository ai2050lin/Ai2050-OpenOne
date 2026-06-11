#!/usr/bin/env bash
set -euo pipefail

cd /home/rankrank/Documents/OpenOne/Ai2050-OpenOne

export OPENONE_NORMAL_ENV="${OPENONE_NORMAL_ENV:-openone-cu130-py312}"
export PHASE86_OUTPUT_DIR="${PHASE86_OUTPUT_DIR:-results/gpt5_phase86_answer_only_reader_calibration_full_$(date +%Y%m%d_%H%M%S)}"
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

mkdir -p "$PHASE86_OUTPUT_DIR"

echo "=== Phase86 answer-only reader calibration full ==="
date '+%Y-%m-%d %H:%M:%S %Z'
echo "conda_env=${CONDA_DEFAULT_ENV:-none}"
echo "output_dir=${PHASE86_OUTPUT_DIR}"

run_one() {
  local model="$1"
  local max_items="$2"
  local attn_impls="${3:-flash_attention_2,sdpa,eager}"
  local out_file="${PHASE86_OUTPUT_DIR}/${model}_phase86_answer_only_reader_calibration.json"
  if [[ -f "$out_file" ]]; then
    echo "=== Skip existing ${model}: ${out_file} ==="
    return
  fi
  echo
  echo "=== Run ${model}: max_items=${max_items}, templates=${PHASE86_TEMPLATES:-all} ==="
  date '+%Y-%m-%d %H:%M:%S %Z'
  PHASE86_ATTN_IMPLEMENTATIONS="$attn_impls" \
    python tests/gpt5/phase86_answer_only_reader_calibration.py "$model" \
      --max-items "$max_items" \
      --relations "${PHASE86_RELATIONS:-}" \
      --frames "${PHASE86_FRAMES:-}" \
      --templates "${PHASE86_TEMPLATES:-}" \
      --max-new-tokens "${PHASE86_MAX_NEW_TOKENS:-8}" \
      --output-dir "$PHASE86_OUTPUT_DIR" \
      --progress-every "${PHASE86_PROGRESS_EVERY:-84}" \
      --attn-implementations "$attn_impls" \
      --hard-exit-after-model
  echo "=== Completed ${model}; process hard-exited ==="
  sleep "${SLEEP_AFTER_MODEL:-3}"
}

IFS=',' read -r -a PHASE86_MODELS_ARRAY <<< "${PHASE86_MODELS:-qwen3,glm4,deepseek7b}"
for model in "${PHASE86_MODELS_ARRAY[@]}"; do
  case "$model" in
    qwen3)
      run_one qwen3 "${QWEN3_PHASE86_MAX_ITEMS:-672}" "${QWEN3_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
      ;;
    glm4)
      run_one glm4 "${GLM4_PHASE86_MAX_ITEMS:-672}" "${GLM4_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
      ;;
    deepseek7b)
      run_one deepseek7b "${DEEPSEEK7B_PHASE86_MAX_ITEMS:-672}" "${DEEPSEEK7B_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
      ;;
    *)
      echo "Unknown model in PHASE86_MODELS: $model" >&2
      exit 2
      ;;
  esac
done

python tests/gpt5/phase86_answer_only_reader_calibration_summary.py \
  --input-dir "$PHASE86_OUTPUT_DIR" \
  --output-dir "$PHASE86_OUTPUT_DIR"

echo
echo "=== Phase86 done ==="
date '+%Y-%m-%d %H:%M:%S %Z'
