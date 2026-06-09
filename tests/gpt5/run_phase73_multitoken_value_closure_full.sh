#!/usr/bin/env bash
set -euo pipefail

cd /home/rankrank/Documents/OpenOne/Ai2050-OpenOne

export OPENONE_NORMAL_ENV="${OPENONE_NORMAL_ENV:-openone-cu130-py312}"
export PHASE73_OUTPUT_DIR="${PHASE73_OUTPUT_DIR:-results/gpt5_phase73_multitoken_value_closure_full_$(date +%Y%m%d_%H%M%S)}"
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

mkdir -p "$PHASE73_OUTPUT_DIR"

echo "=== Phase73 multitoken value full-sequence closure full ==="
date '+%Y-%m-%d %H:%M:%S %Z'
echo "conda_env=${CONDA_DEFAULT_ENV:-none}"
echo "output_dir=${PHASE73_OUTPUT_DIR}"

run_one() {
  local model="$1"
  local layer_pairs="$2"
  local max_items="$3"
  local attn_impls="${4:-flash_attention_2,sdpa,eager}"
  local out_file="${PHASE73_OUTPUT_DIR}/${model}_phase73_multitoken_value_closure.json"
  if [[ -f "$out_file" ]]; then
    echo "=== Skip existing ${model}: ${out_file} ==="
    return
  fi
  echo
  echo "=== Run ${model}: layer_pairs=${layer_pairs}, max_items=${max_items} ==="
  date '+%Y-%m-%d %H:%M:%S %Z'
  PHASE73_ATTN_IMPLEMENTATIONS="$attn_impls" \
    python tests/gpt5/phase73_multitoken_value_closure.py "$model" \
      --layer-pairs "$layer_pairs" \
      --max-items "$max_items" \
      --module "${PHASE73_MODULE:-resid_out}" \
      --positions "${PHASE73_POSITIONS:-object_first,object_last}" \
      --relations "${PHASE73_RELATIONS:-}" \
      --frames "${PHASE73_FRAMES:-}" \
      --output-dir "$PHASE73_OUTPUT_DIR" \
      --progress-every "${PHASE73_PROGRESS_EVERY:-24}" \
      --hard-exit-after-model
  echo "=== Completed ${model}; process hard-exited ==="
  sleep "${SLEEP_AFTER_MODEL:-3}"
}

IFS=',' read -r -a PHASE73_MODELS_ARRAY <<< "${PHASE73_MODELS:-qwen3,glm4,deepseek7b}"
for model in "${PHASE73_MODELS_ARRAY[@]}"; do
  case "$model" in
    qwen3)
      run_one qwen3 "${QWEN3_PHASE73_LAYER_PAIRS:-4-8,8-12,8-16}" "${QWEN3_PHASE73_MAX_ITEMS:-336}" "${QWEN3_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
      ;;
    glm4)
      run_one glm4 "${GLM4_PHASE73_LAYER_PAIRS:-4-10,10-20,4-30}" "${GLM4_PHASE73_MAX_ITEMS:-336}" "${GLM4_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
      ;;
    deepseek7b)
      run_one deepseek7b "${DEEPSEEK7B_PHASE73_LAYER_PAIRS:-8-10,8-12,12-14,12-16}" "${DEEPSEEK7B_PHASE73_MAX_ITEMS:-336}" "${DEEPSEEK7B_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
      ;;
    *)
      echo "Unknown model in PHASE73_MODELS: $model" >&2
      exit 2
      ;;
  esac
done

python tests/gpt5/phase73_multitoken_value_closure_summary.py \
  --input-dir "$PHASE73_OUTPUT_DIR" \
  --output-dir "$PHASE73_OUTPUT_DIR"

echo
echo "=== Phase73 done ==="
date '+%Y-%m-%d %H:%M:%S %Z'
