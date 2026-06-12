#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${PHASE100_OUTPUT_DIR:-results/gpt5_phase100_head_route_restore_source_full_${TS}}"
mkdir -p "$OUT_DIR"

COMMON_ARGS=(
  --slots category,color,function,material,location
  --max-items 210
  --source-attn-items 0
  --positions prompt_tail,last4
  --donor-kind same_slot_diff_target
  --choice-template choice_json_letter
  --progress-every 35
  --output-dir "$OUT_DIR"
  --hard-exit-after-model
)

echo "[phase100] output_dir=$OUT_DIR"

python tests/gpt5/phase100_head_route_restore_source.py qwen3 \
  --layer 24 \
  --head-sets "single29=29;single31=31;pair2931=29,31;wide282931=28,29,31" \
  "${COMMON_ARGS[@]}" 2>&1 | tee "$OUT_DIR/qwen3.log"

python tests/gpt5/phase100_head_route_restore_source.py glm4 \
  --layer 39 \
  --head-sets "single31=31;single17=17;pair3117=31,17;wide311720=31,17,20" \
  "${COMMON_ARGS[@]}" 2>&1 | tee "$OUT_DIR/glm4.log"

python tests/gpt5/phase100_head_route_restore_source.py deepseek7b \
  --layer 27 \
  --head-sets "single21=21;single26=26;pair2126=21,26;wide212609=21,26,9" \
  "${COMMON_ARGS[@]}" 2>&1 | tee "$OUT_DIR/deepseek7b.log"

SOURCE_ARGS=(
  --slots category,color,function,material,location
  --max-items 210
  --source-attn-items 60
  --positions prompt_tail,last4
  --donor-kind same_slot_diff_target
  --choice-template choice_json_letter
  --progress-every 35
  --output-dir "$OUT_DIR"
  --attn-implementations eager
  --source-only
  --hard-exit-after-model
)

python tests/gpt5/phase100_head_route_restore_source.py qwen3 \
  --layer 24 \
  --head-sets "single29=29;single31=31;pair2931=29,31;wide282931=28,29,31" \
  "${SOURCE_ARGS[@]}" 2>&1 | tee "$OUT_DIR/qwen3_source.log"

python tests/gpt5/phase100_head_route_restore_source.py glm4 \
  --layer 39 \
  --head-sets "single31=31;single17=17;pair3117=31,17;wide311720=31,17,20" \
  "${SOURCE_ARGS[@]}" 2>&1 | tee "$OUT_DIR/glm4_source.log"

python tests/gpt5/phase100_head_route_restore_source.py deepseek7b \
  --layer 27 \
  --head-sets "single21=21;single26=26;pair2126=21,26;wide212609=21,26,9" \
  "${SOURCE_ARGS[@]}" 2>&1 | tee "$OUT_DIR/deepseek7b_source.log"

python tests/gpt5/phase100_head_route_restore_source_summary.py --output-dir "$OUT_DIR" | tee "$OUT_DIR/summary.log"

echo "[phase100] done output_dir=$OUT_DIR"
