#!/usr/bin/env bash
set -euo pipefail

cd /home/rankrank/Documents/OpenOne/Ai2050-OpenOne

export OPENONE_NORMAL_ENV="${OPENONE_NORMAL_ENV:-openone-cu130-py312}"
export PHASE78_OUTPUT_DIR="${PHASE78_OUTPUT_DIR:-results/gpt5_phase78_factor_subspace_audit_full_$(date +%Y%m%d_%H%M%S)}"
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

mkdir -p "$PHASE78_OUTPUT_DIR"

echo "=== Phase78 factor subspace audit full ==="
date '+%Y-%m-%d %H:%M:%S %Z'
echo "conda_env=${CONDA_DEFAULT_ENV:-none}"
echo "output_dir=${PHASE78_OUTPUT_DIR}"

run_one() {
  local model="$1"
  local layer_pairs="$2"
  local max_items="$3"
  local attn_impls="${4:-flash_attention_2,sdpa,eager}"
  local out_file="${PHASE78_OUTPUT_DIR}/${model}_phase78_factor_subspace_audit.json"
  if [[ -f "$out_file" ]]; then
    echo "=== Skip existing ${model}: ${out_file} ==="
    return
  fi
  echo
  echo "=== Run ${model}: layer_pairs=${layer_pairs}, max_items=${max_items}, basis_rank=${PHASE78_BASIS_RANK:-16} ==="
  date '+%Y-%m-%d %H:%M:%S %Z'
  PHASE78_ATTN_IMPLEMENTATIONS="$attn_impls" \
    python tests/gpt5/phase78_factor_subspace_audit.py "$model" \
      --layer-pairs "$layer_pairs" \
      --max-items "$max_items" \
      --module "${PHASE78_MODULE:-resid_out}" \
      --relations "${PHASE78_RELATIONS:-}" \
      --frames "${PHASE78_FRAMES:-}" \
      --basis-rank "${PHASE78_BASIS_RANK:-16}" \
      --max-basis-items "${PHASE78_MAX_BASIS_ITEMS:-168}" \
      --output-dir "$PHASE78_OUTPUT_DIR" \
      --progress-every "${PHASE78_PROGRESS_EVERY:-84}" \
      --attn-implementations "$attn_impls" \
      --hard-exit-after-model
  echo "=== Completed ${model}; process hard-exited ==="
  sleep "${SLEEP_AFTER_MODEL:-3}"
}

IFS=',' read -r -a PHASE78_MODELS_ARRAY <<< "${PHASE78_MODELS:-qwen3,glm4,deepseek7b}"
for model in "${PHASE78_MODELS_ARRAY[@]}"; do
  case "$model" in
    qwen3)
      run_one qwen3 "${QWEN3_PHASE78_LAYER_PAIRS:-4-8,8-12}" "${QWEN3_PHASE78_MAX_ITEMS:-672}" "${QWEN3_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
      ;;
    glm4)
      run_one glm4 "${GLM4_PHASE78_LAYER_PAIRS:-4-10,10-20}" "${GLM4_PHASE78_MAX_ITEMS:-672}" "${GLM4_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
      ;;
    deepseek7b)
      run_one deepseek7b "${DEEPSEEK7B_PHASE78_LAYER_PAIRS:-8-10,12-14}" "${DEEPSEEK7B_PHASE78_MAX_ITEMS:-672}" "${DEEPSEEK7B_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
      ;;
    *)
      echo "Unknown model in PHASE78_MODELS: $model" >&2
      exit 2
      ;;
  esac
done

python tests/gpt5/phase78_factor_subspace_audit_summary.py \
  --input-dir "$PHASE78_OUTPUT_DIR" \
  --output-dir "$PHASE78_OUTPUT_DIR"

echo
echo "=== Phase78 done ==="
date '+%Y-%m-%d %H:%M:%S %Z'
