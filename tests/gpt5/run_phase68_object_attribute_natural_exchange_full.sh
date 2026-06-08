#!/usr/bin/env bash
set -euo pipefail

cd /home/rankrank/Documents/OpenOne/Ai2050-OpenOne

export OPENONE_NORMAL_ENV="${OPENONE_NORMAL_ENV:-openone-cu130-py312}"
export PHASE68_OUTPUT_DIR="${PHASE68_OUTPUT_DIR:-results/gpt5_phase68_object_attribute_natural_exchange_full_$(date +%Y%m%d_%H%M%S)}"
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

mkdir -p "$PHASE68_OUTPUT_DIR"

echo "=== Phase68 object-attribute natural exchange full ==="
date '+%Y-%m-%d %H:%M:%S %Z'
echo "conda_env=${CONDA_DEFAULT_ENV:-none}"
echo "output_dir=${PHASE68_OUTPUT_DIR}"

run_one() {
  local model="$1"
  local layers="$2"
  local max_items="$3"
  local attn_impls="${4:-flash_attention_2,sdpa,eager}"
  local out_file="${PHASE68_OUTPUT_DIR}/${model}_phase68_object_attribute_natural_exchange.json"
  if [[ -f "$out_file" ]]; then
    echo "=== Skip existing ${model}: ${out_file} ==="
    return
  fi
  echo
  echo "=== Run ${model}: layers=${layers}, max_items=${max_items} ==="
  date '+%Y-%m-%d %H:%M:%S %Z'
  PHASE68_ATTN_IMPLEMENTATIONS="$attn_impls" \
    python tests/gpt5/phase68_object_attribute_natural_exchange.py "$model" \
      --layers "$layers" \
      --max-items "$max_items" \
      --modules "${PHASE68_MODULES:-resid_out,mlp_out}" \
      --positions "${PHASE68_POSITIONS:-object_first,object_last,last}" \
      --frames "${PHASE68_FRAMES:-the,this,that,a}" \
      --output-dir "$PHASE68_OUTPUT_DIR" \
      --progress-every "${PHASE68_PROGRESS_EVERY:-24}" \
      --hard-exit-after-model
  echo "=== Completed ${model}; process hard-exited ==="
  sleep "${SLEEP_AFTER_MODEL:-3}"
}

run_one qwen3 "${QWEN3_PHASE68_LAYERS:-4,8,12,16}" "${QWEN3_PHASE68_MAX_ITEMS:-192}" "${QWEN3_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
run_one glm4 "${GLM4_PHASE68_LAYERS:-4,10,20,30}" "${GLM4_PHASE68_MAX_ITEMS:-192}" "${GLM4_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
run_one deepseek7b "${DEEPSEEK7B_PHASE68_LAYERS:-8,9,10,11,12,13,14,15,16}" "${DEEPSEEK7B_PHASE68_MAX_ITEMS:-256}" "${DEEPSEEK7B_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"

python tests/gpt5/phase68_object_attribute_natural_exchange_summary.py \
  --input-dir "$PHASE68_OUTPUT_DIR" \
  --output-dir "$PHASE68_OUTPUT_DIR"

echo
echo "=== Phase68 done ==="
date '+%Y-%m-%d %H:%M:%S %Z'
