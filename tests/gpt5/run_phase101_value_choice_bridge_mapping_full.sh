#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${PHASE101_OUTPUT_DIR:-results/gpt5_phase101_value_choice_bridge_mapping_full_${TS}}"
mkdir -p "$OUT_DIR"

COMMON_ARGS=(
  --slots category,color,function,material,location
  --max-items 120
  --choice-position prompt_tail
  --donor-kind same_slot_diff_target
  --choice-template choice_json_letter
  --progress-every 30
  --output-dir "$OUT_DIR"
  --hard-exit-after-model
)

echo "[phase101] output_dir=$OUT_DIR"

python tests/gpt5/phase101_value_choice_bridge_mapping.py qwen3 \
  --value-layer 6 \
  --value-component mlp \
  --value-position prefix8 \
  --choice-layer 24 \
  --choice-heads 29,31 \
  "${COMMON_ARGS[@]}" 2>&1 | tee "$OUT_DIR/qwen3.log"

python tests/gpt5/phase101_value_choice_bridge_mapping.py glm4 \
  --value-layer 39 \
  --value-component mlp \
  --value-position prefix8 \
  --choice-layer 39 \
  --choice-heads 31,17 \
  "${COMMON_ARGS[@]}" 2>&1 | tee "$OUT_DIR/glm4.log"

python tests/gpt5/phase101_value_choice_bridge_mapping.py deepseek7b \
  --value-layer 27 \
  --value-component mlp \
  --value-position prefix8 \
  --choice-layer 27 \
  --choice-heads 21,26 \
  "${COMMON_ARGS[@]}" 2>&1 | tee "$OUT_DIR/deepseek7b.log"

python tests/gpt5/phase101_value_choice_bridge_mapping_summary.py --output-dir "$OUT_DIR" | tee "$OUT_DIR/summary.log"

echo "[phase101] done output_dir=$OUT_DIR"
