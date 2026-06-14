#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${PHASE102_OUTPUT_DIR:-results/gpt5_phase102_value_factor_bridge_decomposition_full_${TS}}"
mkdir -p "$OUT_DIR"

COMMON_ARGS=(
  --slots category,color,function,material,location
  --max-items "${PHASE102_MAX_ITEMS:-240}"
  --choice-position prompt_tail
  --choice-template choice_json_letter
  --rank "${PHASE102_RANK:-4}"
  --pool-mode prefix
  --copy-mode both
  --progress-every "${PHASE102_PROGRESS_EVERY:-40}"
  --output-dir "$OUT_DIR"
  --hard-exit-after-model
)

echo "[phase102] output_dir=$OUT_DIR"

python tests/gpt5/phase102_value_factor_bridge_decomposition.py qwen3 \
  --value-layer 6 \
  --value-component mlp \
  --value-position prefix8 \
  --choice-layer 24 \
  --choice-heads 29,31 \
  "${COMMON_ARGS[@]}" 2>&1 | tee "$OUT_DIR/qwen3.log"

python tests/gpt5/phase102_value_factor_bridge_decomposition.py glm4 \
  --value-layer 39 \
  --value-component mlp \
  --value-position prefix8 \
  --choice-layer 39 \
  --choice-heads 31,17 \
  "${COMMON_ARGS[@]}" 2>&1 | tee "$OUT_DIR/glm4.log"

python tests/gpt5/phase102_value_factor_bridge_decomposition.py deepseek7b \
  --value-layer 27 \
  --value-component mlp \
  --value-position prefix8 \
  --choice-layer 27 \
  --choice-heads 21,26 \
  "${COMMON_ARGS[@]}" 2>&1 | tee "$OUT_DIR/deepseek7b.log"

python tests/gpt5/phase102_value_factor_bridge_decomposition_summary.py --output-dir "$OUT_DIR" | tee "$OUT_DIR/summary.log"

echo "[phase102] done output_dir=$OUT_DIR"
