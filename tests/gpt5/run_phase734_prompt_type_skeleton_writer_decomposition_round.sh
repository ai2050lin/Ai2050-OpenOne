#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${PHASE734_ROUND_NAME:-main}"
PHASE733_ROUND="${PHASE734_PHASE733_ROUND:-confirm}"
MAX_PAIRS="${PHASE734_MAX_PAIRS:-12}"
MAX_LAYERS="${PHASE734_MAX_LAYERS:-}"
MAX_HEADS_PER_LAYER="${PHASE734_MAX_HEADS_PER_LAYER:-12}"
MLP_GROUPS_PER_LAYER="${PHASE734_MLP_GROUPS_PER_LAYER:-12}"
LOG_EVERY="${PHASE734_LOG_EVERY:-4}"

OUT_DIR="results/glm5_phase734_prompt_type_skeleton_writer_decomposition/${ROUND_NAME}"
mkdir -p "${OUT_DIR}"

echo "[phase734] round=${ROUND_NAME} phase733_round=${PHASE733_ROUND} pairs=${MAX_PAIRS} heads_per_layer=${MAX_HEADS_PER_LAYER} mlp_groups=${MLP_GROUPS_PER_LAYER}"

for MODEL in qwen3 glm4 deepseek7b; do
  echo "[phase734] start model=${MODEL} at $(date '+%Y-%m-%d %H:%M:%S')"
  CMD=(
    python tests/gpt5/phase734_prompt_type_skeleton_writer_decomposition.py
    --model "${MODEL}"
    --round-name "${ROUND_NAME}"
    --phase733-round "${PHASE733_ROUND}"
    --max-pairs "${MAX_PAIRS}"
    --max-heads-per-layer "${MAX_HEADS_PER_LAYER}"
    --mlp-groups-per-layer "${MLP_GROUPS_PER_LAYER}"
    --log-every "${LOG_EVERY}"
    --hard-exit-after-model
  )
  if [[ -n "${MAX_LAYERS}" ]]; then
    CMD+=(--max-layers "${MAX_LAYERS}")
  fi
  "${CMD[@]}" 2>&1 | tee "${OUT_DIR}/phase734_${MODEL}.log"
  echo "[phase734] done model=${MODEL} at $(date '+%Y-%m-%d %H:%M:%S')"
done

python tests/gpt5/phase734_prompt_type_skeleton_writer_decomposition.py \
  --round-name "${ROUND_NAME}" \
  --summarize-only 2>&1 | tee "${OUT_DIR}/phase734_summary.log"

echo "[phase734] round=${ROUND_NAME} complete at $(date '+%Y-%m-%d %H:%M:%S')"
