#!/usr/bin/env bash
set -euo pipefail

cd /home/rankrank/Documents/OpenOne/Ai2050-OpenOne

export OPENONE_NORMAL_ENV="${OPENONE_NORMAL_ENV:-openone-cu130-py312}"
export PHASE67_OUTPUT_DIR="${PHASE67_OUTPUT_DIR:-results/gpt5_phase67_object_attribute_path_full_$(date +%Y%m%d_%H%M%S)}"
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

mkdir -p "$PHASE67_OUTPUT_DIR"

echo "=== Phase67 object-attribute path full ==="
date '+%Y-%m-%d %H:%M:%S %Z'
echo "conda_env=${CONDA_DEFAULT_ENV:-none}"
echo "output_dir=${PHASE67_OUTPUT_DIR}"

run_one() {
  local model="$1"
  local layers="$2"
  local max_items="$3"
  local attn_impls="${4:-flash_attention_2,sdpa,eager}"
  local out_file="${PHASE67_OUTPUT_DIR}/${model}_phase67_object_attribute_path_map.json"
  if [[ -f "$out_file" ]]; then
    echo "=== Skip existing ${model}: ${out_file} ==="
    return
  fi
  echo
  echo "=== Run ${model}: layers=${layers}, max_items=${max_items} ==="
  date '+%Y-%m-%d %H:%M:%S %Z'
  PHASE67_ATTN_IMPLEMENTATIONS="$attn_impls" \
    python tests/gpt5/phase67_object_attribute_path_map.py "$model" \
      --layers "$layers" \
      --max-items "$max_items" \
      --modules "${PHASE67_MODULES:-resid_out,attn_out,mlp_out}" \
      --positions "${PHASE67_POSITIONS:-object_first,object_last,last}" \
      --frames "${PHASE67_FRAMES:-the,this,that,a}" \
      --output-dir "$PHASE67_OUTPUT_DIR" \
      --progress-every "${PHASE67_PROGRESS_EVERY:-12}" \
      --hard-exit-after-model
  echo "=== Completed ${model}; process hard-exited ==="
  sleep "${SLEEP_AFTER_MODEL:-3}"
}

run_one qwen3 "${QWEN3_PHASE67_LAYERS:-4,8,12,16,20}" "${QWEN3_PHASE67_MAX_ITEMS:-72}" "${QWEN3_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
run_one glm4 "${GLM4_PHASE67_LAYERS:-4,10,20,30}" "${GLM4_PHASE67_MAX_ITEMS:-72}" "${GLM4_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
run_one deepseek7b "${DEEPSEEK7B_PHASE67_LAYERS:-8,12,16,20}" "${DEEPSEEK7B_PHASE67_MAX_ITEMS:-144}" "${DEEPSEEK7B_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"

python tests/gpt5/phase67_object_attribute_path_summary.py \
  --input-dir "$PHASE67_OUTPUT_DIR" \
  --output-dir "$PHASE67_OUTPUT_DIR"

echo
echo "=== Phase67 done ==="
date '+%Y-%m-%d %H:%M:%S %Z'
