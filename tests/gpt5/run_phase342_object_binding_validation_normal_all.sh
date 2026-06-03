#!/usr/bin/env bash
set -euo pipefail

cd /home/rankrank/Documents/OpenOne/Ai2050-OpenOne

export OPENONE_NORMAL_ENV="${OPENONE_NORMAL_ENV:-openone-cu130-py312}"
export PHASE339_OUTPUT_DIR="${PHASE339_OUTPUT_DIR:-results/gpt5_phase342_object_binding_validation_full}"
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

mkdir -p "$PHASE339_OUTPUT_DIR"

echo "=== Phase342 object-property binding validation normal all-model ==="
date '+%Y-%m-%d %H:%M:%S %Z'
echo "conda_env=${CONDA_DEFAULT_ENV:-none}"
echo "output_dir=${PHASE339_OUTPUT_DIR}"
echo "nvidia_driver=$(cat /proc/driver/nvidia/version 2>/dev/null | head -n 1 || true)"
echo

run_model() {
  local model="$1"
  echo
  echo "=== Run ${model} ==="
  date '+%Y-%m-%d %H:%M:%S %Z'
  python tests/glm5/phase339_multibaseline_pipeline.py "$model" \
    --output-dir "$PHASE339_OUTPUT_DIR" \
    --hard-exit-after-model
  echo "=== Completed ${model}; process hard-exited ==="
  sleep "${SLEEP_AFTER_MODEL:-5}"
}

run_model qwen3
run_model glm4
run_model deepseek7b

python tests/gpt5/phase342_object_binding_validation_summary.py \
  --input-dir "$PHASE339_OUTPUT_DIR" \
  --output-dir "$PHASE339_OUTPUT_DIR"

echo
echo "=== Phase342 done ==="
date '+%Y-%m-%d %H:%M:%S %Z'
