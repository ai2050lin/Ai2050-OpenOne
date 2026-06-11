#!/usr/bin/env bash
set -euo pipefail

cd /home/rankrank/Documents/OpenOne/Ai2050-OpenOne

export OPENONE_NORMAL_ENV="${OPENONE_NORMAL_ENV:-openone-cu130-py312}"
export PHASE87_OUTPUT_DIR="${PHASE87_OUTPUT_DIR:-results/gpt5_phase87_reader_stack_calibration_full_$(date +%Y%m%d_%H%M%S)}"
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

mkdir -p "$PHASE87_OUTPUT_DIR"

echo "=== Phase87 reader stack calibration full ==="
date '+%Y-%m-%d %H:%M:%S %Z'
echo "conda_env=${CONDA_DEFAULT_ENV:-none}"
echo "output_dir=${PHASE87_OUTPUT_DIR}"

run_one() {
  local model="$1"
  local max_items="$2"
  local attn_impls="${3:-flash_attention_2,sdpa,eager}"
  local out_file="${PHASE87_OUTPUT_DIR}/${model}_phase87_reader_stack_calibration.json"
  if [[ -f "$out_file" ]]; then
    echo "=== Skip existing ${model}: ${out_file} ==="
    return
  fi
  echo
  echo "=== Run ${model}: max_items=${max_items} ==="
  date '+%Y-%m-%d %H:%M:%S %Z'
  PHASE87_ATTN_IMPLEMENTATIONS="$attn_impls" \
    python tests/gpt5/phase87_reader_stack_calibration.py "$model" \
      --max-items "$max_items" \
      --max-distractors "${PHASE87_MAX_DISTRACTORS:-4}" \
      --relations "${PHASE87_RELATIONS:-}" \
      --frames "${PHASE87_FRAMES:-}" \
      --choice-templates "${PHASE87_CHOICE_TEMPLATES:-}" \
      --open-templates "${PHASE87_OPEN_TEMPLATES:-}" \
      --choice-max-new-tokens "${PHASE87_CHOICE_MAX_NEW_TOKENS:-4}" \
      --open-max-new-tokens "${PHASE87_OPEN_MAX_NEW_TOKENS:-8}" \
      --output-dir "$PHASE87_OUTPUT_DIR" \
      --progress-every "${PHASE87_PROGRESS_EVERY:-84}" \
      --attn-implementations "$attn_impls" \
      --hard-exit-after-model
  echo "=== Completed ${model}; process hard-exited ==="
  sleep "${SLEEP_AFTER_MODEL:-3}"
}

IFS=',' read -r -a PHASE87_MODELS_ARRAY <<< "${PHASE87_MODELS:-qwen3,glm4,deepseek7b}"
for model in "${PHASE87_MODELS_ARRAY[@]}"; do
  case "$model" in
    qwen3)
      run_one qwen3 "${QWEN3_PHASE87_MAX_ITEMS:-672}" "${QWEN3_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
      ;;
    glm4)
      run_one glm4 "${GLM4_PHASE87_MAX_ITEMS:-672}" "${GLM4_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
      ;;
    deepseek7b)
      run_one deepseek7b "${DEEPSEEK7B_PHASE87_MAX_ITEMS:-672}" "${DEEPSEEK7B_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
      ;;
    *)
      echo "Unknown model in PHASE87_MODELS: $model" >&2
      exit 2
      ;;
  esac
done

python tests/gpt5/phase87_reader_stack_calibration_summary.py \
  --input-dir "$PHASE87_OUTPUT_DIR" \
  --output-dir "$PHASE87_OUTPUT_DIR"

echo
echo "=== Phase87 done ==="
date '+%Y-%m-%d %H:%M:%S %Z'
