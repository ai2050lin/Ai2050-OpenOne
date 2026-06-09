#!/usr/bin/env bash
set -euo pipefail

cd /home/rankrank/Documents/OpenOne/Ai2050-OpenOne

export OPENONE_NORMAL_ENV="${OPENONE_NORMAL_ENV:-openone-cu130-py312}"
export PHASE72_OUTPUT_DIR="${PHASE72_OUTPUT_DIR:-results/gpt5_phase72_object_relation_value_fullseq_closure_full_$(date +%Y%m%d_%H%M%S)}"
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

mkdir -p "$PHASE72_OUTPUT_DIR"

echo "=== Phase72 object-relation-value full-sequence closure full ==="
date '+%Y-%m-%d %H:%M:%S %Z'
echo "conda_env=${CONDA_DEFAULT_ENV:-none}"
echo "output_dir=${PHASE72_OUTPUT_DIR}"

run_one() {
  local model="$1"
  local layer_pairs="$2"
  local max_items="$3"
  local attn_impls="${4:-flash_attention_2,sdpa,eager}"
  local out_file="${PHASE72_OUTPUT_DIR}/${model}_phase72_object_relation_value_fullseq_closure.json"
  if [[ -f "$out_file" ]]; then
    echo "=== Skip existing ${model}: ${out_file} ==="
    return
  fi
  echo
  echo "=== Run ${model}: layer_pairs=${layer_pairs}, max_items=${max_items} ==="
  date '+%Y-%m-%d %H:%M:%S %Z'
  PHASE72_ATTN_IMPLEMENTATIONS="$attn_impls" \
    python tests/gpt5/phase72_object_relation_value_fullseq_closure.py "$model" \
      --layer-pairs "$layer_pairs" \
      --max-items "$max_items" \
      --module "${PHASE72_MODULE:-resid_out}" \
      --positions "${PHASE72_POSITIONS:-object_first,object_last}" \
      --relations "${PHASE72_RELATIONS:-}" \
      --frames "${PHASE72_FRAMES:-}" \
      --output-dir "$PHASE72_OUTPUT_DIR" \
      --progress-every "${PHASE72_PROGRESS_EVERY:-24}" \
      --hard-exit-after-model
  echo "=== Completed ${model}; process hard-exited ==="
  sleep "${SLEEP_AFTER_MODEL:-3}"
}

run_one qwen3 "${QWEN3_PHASE72_LAYER_PAIRS:-4-8,8-12,8-16}" "${QWEN3_PHASE72_MAX_ITEMS:-342}" "${QWEN3_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
run_one glm4 "${GLM4_PHASE72_LAYER_PAIRS:-4-10,10-20,4-30}" "${GLM4_PHASE72_MAX_ITEMS:-342}" "${GLM4_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
run_one deepseek7b "${DEEPSEEK7B_PHASE72_LAYER_PAIRS:-8-10,8-12,12-14,12-16}" "${DEEPSEEK7B_PHASE72_MAX_ITEMS:-342}" "${DEEPSEEK7B_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"

python tests/gpt5/phase72_object_relation_value_fullseq_closure_summary.py \
  --input-dir "$PHASE72_OUTPUT_DIR" \
  --output-dir "$PHASE72_OUTPUT_DIR"

echo
echo "=== Phase72 done ==="
date '+%Y-%m-%d %H:%M:%S %Z'
