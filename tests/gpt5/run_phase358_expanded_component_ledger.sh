#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export OPENONE_NORMAL_ENV="${OPENONE_NORMAL_ENV:-openone-cu130-py312}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "$OPENONE_NORMAL_ENV" ]]; then
  if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
  else
    source /home/rankrank/miniconda3/etc/profile.d/conda.sh
  fi
  conda activate "$OPENONE_NORMAL_ENV"
fi

for stage in blind_discovery blind_calibration; do
  for model in qwen3 glm4 deepseek7b; do
    python tests/gpt5/phase358_multiresolution_component_conservation.py --model "$model" --stage "$stage"
  done
done
python tests/gpt5/phase358_expanded_ledger_analysis.py
