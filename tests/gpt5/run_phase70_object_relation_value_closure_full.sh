#!/usr/bin/env bash
set -euo pipefail

cd /home/rankrank/Documents/OpenOne/Ai2050-OpenOne

export OPENONE_NORMAL_ENV="${OPENONE_NORMAL_ENV:-openone-cu130-py312}"
export PHASE70_OUTPUT_DIR="${PHASE70_OUTPUT_DIR:-results/gpt5_phase70_object_relation_value_closure_full_$(date +%Y%m%d_%H%M%S)}"
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

mkdir -p "$PHASE70_OUTPUT_DIR"

echo "=== Phase70 object-relation-value closure full ==="
date '+%Y-%m-%d %H:%M:%S %Z'
echo "conda_env=${CONDA_DEFAULT_ENV:-none}"
echo "output_dir=${PHASE70_OUTPUT_DIR}"

run_one() {
  local model="$1"
  local layer_pairs="$2"
  local max_items="$3"
  local attn_impls="${4:-flash_attention_2,sdpa,eager}"
  local out_file="${PHASE70_OUTPUT_DIR}/${model}_phase70_object_relation_value_closure.json"
  if [[ -f "$out_file" ]]; then
    echo "=== Skip existing ${model}: ${out_file} ==="
    return
  fi
  echo
  echo "=== Run ${model}: layer_pairs=${layer_pairs}, max_items=${max_items} ==="
  date '+%Y-%m-%d %H:%M:%S %Z'
  PHASE70_ATTN_IMPLEMENTATIONS="$attn_impls" \
    python tests/gpt5/phase70_object_relation_value_closure.py "$model" \
      --layer-pairs "$layer_pairs" \
      --max-items "$max_items" \
      --module "${PHASE70_MODULE:-resid_out}" \
      --positions "${PHASE70_POSITIONS:-object_first,object_last,last}" \
      --relations "${PHASE70_RELATIONS:-}" \
      --frames "${PHASE70_FRAMES:-}" \
      --output-dir "$PHASE70_OUTPUT_DIR" \
      --progress-every "${PHASE70_PROGRESS_EVERY:-32}" \
      --hard-exit-after-model
  echo "=== Completed ${model}; process hard-exited ==="
  sleep "${SLEEP_AFTER_MODEL:-3}"
}

run_one qwen3 "${QWEN3_PHASE70_LAYER_PAIRS:-4-8,4-12,4-16,8-12,8-16}" "${QWEN3_PHASE70_MAX_ITEMS:-360}" "${QWEN3_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
run_one glm4 "${GLM4_PHASE70_LAYER_PAIRS:-4-10,4-20,4-30,10-20,10-30,20-30}" "${GLM4_PHASE70_MAX_ITEMS:-360}" "${GLM4_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
run_one deepseek7b "${DEEPSEEK7B_PHASE70_LAYER_PAIRS:-8-10,8-12,8-14,8-16,10-12,10-14,10-16,12-14,12-16}" "${DEEPSEEK7B_PHASE70_MAX_ITEMS:-360}" "${DEEPSEEK7B_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"

python tests/gpt5/phase70_object_relation_value_closure_summary.py \
  --input-dir "$PHASE70_OUTPUT_DIR" \
  --output-dir "$PHASE70_OUTPUT_DIR"

echo
echo "=== Phase70 done ==="
date '+%Y-%m-%d %H:%M:%S %Z'
