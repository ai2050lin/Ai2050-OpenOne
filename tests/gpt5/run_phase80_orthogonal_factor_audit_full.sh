#!/usr/bin/env bash
set -euo pipefail

cd /home/rankrank/Documents/OpenOne/Ai2050-OpenOne

export OPENONE_NORMAL_ENV="${OPENONE_NORMAL_ENV:-openone-cu130-py312}"
export PHASE80_OUTPUT_DIR="${PHASE80_OUTPUT_DIR:-results/gpt5_phase80_orthogonal_factor_audit_full_$(date +%Y%m%d_%H%M%S)}"
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

mkdir -p "$PHASE80_OUTPUT_DIR"

echo "=== Phase80 orthogonal factor audit full ==="
date '+%Y-%m-%d %H:%M:%S %Z'
echo "conda_env=${CONDA_DEFAULT_ENV:-none}"
echo "output_dir=${PHASE80_OUTPUT_DIR}"

run_one() {
  local model="$1"
  local layer_pairs="$2"
  local max_items="$3"
  local attn_impls="${4:-flash_attention_2,sdpa,eager}"
  local out_file="${PHASE80_OUTPUT_DIR}/${model}_phase80_orthogonal_factor_audit.json"
  if [[ -f "$out_file" ]]; then
    echo "=== Skip existing ${model}: ${out_file} ==="
    return
  fi
  echo
  echo "=== Run ${model}: layer_pairs=${layer_pairs}, max_items=${max_items}, contrast_rank=${PHASE80_CONTRAST_RANK:-64}, nuisance_rank=${PHASE80_NUISANCE_RANK:-24} ==="
  date '+%Y-%m-%d %H:%M:%S %Z'
  PHASE80_ATTN_IMPLEMENTATIONS="$attn_impls" \
    python tests/gpt5/phase80_orthogonal_factor_audit.py "$model" \
      --layer-pairs "$layer_pairs" \
      --max-items "$max_items" \
      --module "${PHASE80_MODULE:-resid_out}" \
      --relations "${PHASE80_RELATIONS:-}" \
      --frames "${PHASE80_FRAMES:-}" \
      --contrast-rank "${PHASE80_CONTRAST_RANK:-64}" \
      --nuisance-rank "${PHASE80_NUISANCE_RANK:-24}" \
      --max-basis-items "${PHASE80_MAX_BASIS_ITEMS:-224}" \
      --max-distractors "${PHASE80_MAX_DISTRACTORS:-10}" \
      --output-dir "$PHASE80_OUTPUT_DIR" \
      --progress-every "${PHASE80_PROGRESS_EVERY:-84}" \
      --attn-implementations "$attn_impls" \
      --hard-exit-after-model
  echo "=== Completed ${model}; process hard-exited ==="
  sleep "${SLEEP_AFTER_MODEL:-3}"
}

IFS=',' read -r -a PHASE80_MODELS_ARRAY <<< "${PHASE80_MODELS:-qwen3,glm4,deepseek7b}"
for model in "${PHASE80_MODELS_ARRAY[@]}"; do
  case "$model" in
    qwen3)
      run_one qwen3 "${QWEN3_PHASE80_LAYER_PAIRS:-4-8,8-12}" "${QWEN3_PHASE80_MAX_ITEMS:-672}" "${QWEN3_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
      ;;
    glm4)
      run_one glm4 "${GLM4_PHASE80_LAYER_PAIRS:-4-10,10-20}" "${GLM4_PHASE80_MAX_ITEMS:-672}" "${GLM4_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
      ;;
    deepseek7b)
      run_one deepseek7b "${DEEPSEEK7B_PHASE80_LAYER_PAIRS:-8-10,12-14}" "${DEEPSEEK7B_PHASE80_MAX_ITEMS:-672}" "${DEEPSEEK7B_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
      ;;
    *)
      echo "Unknown model in PHASE80_MODELS: $model" >&2
      exit 2
      ;;
  esac
done

python tests/gpt5/phase80_orthogonal_factor_audit_summary.py \
  --input-dir "$PHASE80_OUTPUT_DIR" \
  --output-dir "$PHASE80_OUTPUT_DIR"

echo
echo "=== Phase80 done ==="
date '+%Y-%m-%d %H:%M:%S %Z'
