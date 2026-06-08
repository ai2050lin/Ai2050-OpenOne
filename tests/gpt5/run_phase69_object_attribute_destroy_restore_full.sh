#!/usr/bin/env bash
set -euo pipefail

cd /home/rankrank/Documents/OpenOne/Ai2050-OpenOne

export OPENONE_NORMAL_ENV="${OPENONE_NORMAL_ENV:-openone-cu130-py312}"
export PHASE69_OUTPUT_DIR="${PHASE69_OUTPUT_DIR:-results/gpt5_phase69_object_attribute_destroy_restore_full_$(date +%Y%m%d_%H%M%S)}"
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

mkdir -p "$PHASE69_OUTPUT_DIR"

echo "=== Phase69 object-attribute destroy-restore full ==="
date '+%Y-%m-%d %H:%M:%S %Z'
echo "conda_env=${CONDA_DEFAULT_ENV:-none}"
echo "output_dir=${PHASE69_OUTPUT_DIR}"

run_one() {
  local model="$1"
  local layer_pairs="$2"
  local max_items="$3"
  local attn_impls="${4:-flash_attention_2,sdpa,eager}"
  local out_file="${PHASE69_OUTPUT_DIR}/${model}_phase69_object_attribute_destroy_restore.json"
  if [[ -f "$out_file" ]]; then
    echo "=== Skip existing ${model}: ${out_file} ==="
    return
  fi
  echo
  echo "=== Run ${model}: layer_pairs=${layer_pairs}, max_items=${max_items} ==="
  date '+%Y-%m-%d %H:%M:%S %Z'
  PHASE69_ATTN_IMPLEMENTATIONS="$attn_impls" \
    python tests/gpt5/phase69_object_attribute_destroy_restore.py "$model" \
      --layer-pairs "$layer_pairs" \
      --max-items "$max_items" \
      --module "${PHASE69_MODULE:-resid_out}" \
      --positions "${PHASE69_POSITIONS:-object_first,object_last,last}" \
      --frames "${PHASE69_FRAMES:-the,this,that,a}" \
      --output-dir "$PHASE69_OUTPUT_DIR" \
      --progress-every "${PHASE69_PROGRESS_EVERY:-32}" \
      --hard-exit-after-model
  echo "=== Completed ${model}; process hard-exited ==="
  sleep "${SLEEP_AFTER_MODEL:-3}"
}

run_one qwen3 "${QWEN3_PHASE69_LAYER_PAIRS:-4-8,4-12,4-16,8-12,8-16}" "${QWEN3_PHASE69_MAX_ITEMS:-192}" "${QWEN3_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
run_one glm4 "${GLM4_PHASE69_LAYER_PAIRS:-4-10,4-20,4-30,10-20,10-30,20-30}" "${GLM4_PHASE69_MAX_ITEMS:-192}" "${GLM4_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
run_one deepseek7b "${DEEPSEEK7B_PHASE69_LAYER_PAIRS:-8-10,8-12,8-14,8-16,10-12,10-14,10-16,12-14,12-16}" "${DEEPSEEK7B_PHASE69_MAX_ITEMS:-248}" "${DEEPSEEK7B_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"

python tests/gpt5/phase69_object_attribute_destroy_restore_summary.py \
  --input-dir "$PHASE69_OUTPUT_DIR" \
  --output-dir "$PHASE69_OUTPUT_DIR"

echo
echo "=== Phase69 done ==="
date '+%Y-%m-%d %H:%M:%S %Z'
