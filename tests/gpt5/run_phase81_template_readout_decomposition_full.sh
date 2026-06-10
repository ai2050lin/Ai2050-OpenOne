#!/usr/bin/env bash
set -euo pipefail

cd /home/rankrank/Documents/OpenOne/Ai2050-OpenOne

export OPENONE_NORMAL_ENV="${OPENONE_NORMAL_ENV:-openone-cu130-py312}"
export PHASE81_OUTPUT_DIR="${PHASE81_OUTPUT_DIR:-results/gpt5_phase81_template_readout_decomposition_full_$(date +%Y%m%d_%H%M%S)}"
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

mkdir -p "$PHASE81_OUTPUT_DIR"

echo "=== Phase81 template/readout decomposition full ==="
date '+%Y-%m-%d %H:%M:%S %Z'
echo "conda_env=${CONDA_DEFAULT_ENV:-none}"
echo "output_dir=${PHASE81_OUTPUT_DIR}"

run_one() {
  local model="$1"
  local layer_pairs="$2"
  local max_items="$3"
  local attn_impls="${4:-flash_attention_2,sdpa,eager}"
  local out_file="${PHASE81_OUTPUT_DIR}/${model}_phase81_template_readout_decomposition.json"
  if [[ -f "$out_file" ]]; then
    echo "=== Skip existing ${model}: ${out_file} ==="
    return
  fi
  echo
  echo "=== Run ${model}: layer_pairs=${layer_pairs}, max_items=${max_items}, contrast_rank=${PHASE81_CONTRAST_RANK:-64}, nuisance_rank=${PHASE81_NUISANCE_RANK:-24} ==="
  date '+%Y-%m-%d %H:%M:%S %Z'
  PHASE81_ATTN_IMPLEMENTATIONS="$attn_impls" \
    python tests/gpt5/phase81_template_readout_decomposition.py "$model" \
      --layer-pairs "$layer_pairs" \
      --max-items "$max_items" \
      --module "${PHASE81_MODULE:-resid_out}" \
      --relations "${PHASE81_RELATIONS:-}" \
      --phrase-ids "${PHASE81_PHRASE_IDS:-}" \
      --slot-ids "${PHASE81_SLOT_IDS:-}" \
      --contrast-rank "${PHASE81_CONTRAST_RANK:-64}" \
      --nuisance-rank "${PHASE81_NUISANCE_RANK:-24}" \
      --max-basis-items "${PHASE81_MAX_BASIS_ITEMS:-448}" \
      --max-distractors "${PHASE81_MAX_DISTRACTORS:-10}" \
      --output-dir "$PHASE81_OUTPUT_DIR" \
      --progress-every "${PHASE81_PROGRESS_EVERY:-168}" \
      --attn-implementations "$attn_impls" \
      --hard-exit-after-model
  echo "=== Completed ${model}; process hard-exited ==="
  sleep "${SLEEP_AFTER_MODEL:-3}"
}

IFS=',' read -r -a PHASE81_MODELS_ARRAY <<< "${PHASE81_MODELS:-qwen3,glm4,deepseek7b}"
for model in "${PHASE81_MODELS_ARRAY[@]}"; do
  case "$model" in
    qwen3)
      run_one qwen3 "${QWEN3_PHASE81_LAYER_PAIRS:-4-8,8-12}" "${QWEN3_PHASE81_MAX_ITEMS:-1344}" "${QWEN3_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
      ;;
    glm4)
      run_one glm4 "${GLM4_PHASE81_LAYER_PAIRS:-4-10,10-20}" "${GLM4_PHASE81_MAX_ITEMS:-1344}" "${GLM4_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
      ;;
    deepseek7b)
      run_one deepseek7b "${DEEPSEEK7B_PHASE81_LAYER_PAIRS:-8-10,12-14}" "${DEEPSEEK7B_PHASE81_MAX_ITEMS:-1344}" "${DEEPSEEK7B_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"
      ;;
    *)
      echo "Unknown model in PHASE81_MODELS: $model" >&2
      exit 2
      ;;
  esac
done

python tests/gpt5/phase81_template_readout_decomposition_summary.py \
  --input-dir "$PHASE81_OUTPUT_DIR" \
  --output-dir "$PHASE81_OUTPUT_DIR"

echo
echo "=== Phase81 done ==="
date '+%Y-%m-%d %H:%M:%S %Z'
