#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${PHASE103_OUTPUT_DIR:-results/gpt5_phase103_bridge_localization_restore_sweep_full_${TS}}"
mkdir -p "$OUT_DIR"

COMMON_ARGS=(
  --slots "${PHASE103_SLOTS:-category,function,location}"
  --max-items "${PHASE103_MAX_ITEMS:-180}"
  --factors "${PHASE103_FACTORS:-value_all,own}"
  --rank "${PHASE103_RANK:-4}"
  --pool-mode prefix
  --choice-position prompt_tail
  --choice-template choice_json_letter
  --progress-every "${PHASE103_PROGRESS_EVERY:-10}"
  --output-dir "$OUT_DIR"
  --hard-exit-after-model
)

echo "[phase103] output_dir=$OUT_DIR"

python tests/gpt5/phase103_bridge_localization_restore_sweep.py qwen3 \
  --value-layer 6 \
  --value-component mlp \
  --value-position prefix8 \
  --restore-nodes "${PHASE103_QWEN_RESTORE_NODES:-8:attn,8:mlp,12:attn,12:mlp,16:attn,16:mlp,20:attn,20:mlp,22:attn,22:mlp,24:attn,24:mlp,24:choice_heads}" \
  --choice-heads 29,31 \
  "${COMMON_ARGS[@]}" 2>&1 | tee "$OUT_DIR/qwen3.log"

python tests/gpt5/phase103_bridge_localization_restore_sweep.py glm4 \
  --value-layer 33 \
  --value-component mlp \
  --value-position prefix8 \
  --restore-nodes "${PHASE103_GLM4_RESTORE_NODES:-35:attn,35:mlp,37:attn,37:mlp,39:attn,39:mlp,39:choice_heads}" \
  --choice-heads 31,17 \
  "${COMMON_ARGS[@]}" 2>&1 | tee "$OUT_DIR/glm4.log"

python tests/gpt5/phase103_bridge_localization_restore_sweep.py deepseek7b \
  --value-layer 24 \
  --value-component mlp \
  --value-position prefix8 \
  --restore-nodes "${PHASE103_DS7B_RESTORE_NODES:-25:attn,25:mlp,26:attn,26:mlp,27:attn,27:mlp,27:choice_heads}" \
  --choice-heads 21,26 \
  "${COMMON_ARGS[@]}" 2>&1 | tee "$OUT_DIR/deepseek7b.log"

python tests/gpt5/phase103_bridge_localization_restore_sweep_summary.py --output-dir "$OUT_DIR" | tee "$OUT_DIR/summary.log"

echo "[phase103] done output_dir=$OUT_DIR"
