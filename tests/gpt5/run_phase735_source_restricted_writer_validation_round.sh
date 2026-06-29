#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${PHASE735_ROUND_NAME:-main}"
PHASE734_ROUND="${PHASE735_PHASE734_ROUND:-confirm}"
MAX_PAIRS="${PHASE735_MAX_PAIRS:-12}"
TOP_ATTN="${PHASE735_TOP_ATTN:-2}"
TOP_MLP="${PHASE735_TOP_MLP:-2}"
MLP_SUBGROUPS="${PHASE735_MLP_SUBGROUPS:-4}"
LOG_EVERY="${PHASE735_LOG_EVERY:-2}"

OUT_DIR="results/glm5_phase735_source_restricted_writer_validation/${ROUND_NAME}"
mkdir -p "${OUT_DIR}"

echo "[phase735] round=${ROUND_NAME} phase734_round=${PHASE734_ROUND} pairs=${MAX_PAIRS} top_attn=${TOP_ATTN} top_mlp=${TOP_MLP} mlp_subgroups=${MLP_SUBGROUPS}"

for MODEL in qwen3 glm4 deepseek7b; do
  echo "[phase735] start model=${MODEL} at $(date '+%Y-%m-%d %H:%M:%S')"
  python tests/gpt5/phase735_source_restricted_writer_validation.py \
    --model "${MODEL}" \
    --round-name "${ROUND_NAME}" \
    --phase734-round "${PHASE734_ROUND}" \
    --max-pairs "${MAX_PAIRS}" \
    --top-attn "${TOP_ATTN}" \
    --top-mlp "${TOP_MLP}" \
    --mlp-subgroups "${MLP_SUBGROUPS}" \
    --log-every "${LOG_EVERY}" \
    --hard-exit-after-model 2>&1 | tee "${OUT_DIR}/phase735_${MODEL}.log"
  echo "[phase735] done model=${MODEL} at $(date '+%Y-%m-%d %H:%M:%S')"
done

python tests/gpt5/phase735_source_restricted_writer_validation.py \
  --round-name "${ROUND_NAME}" \
  --summarize-only 2>&1 | tee "${OUT_DIR}/phase735_summary.log"

echo "[phase735] round=${ROUND_NAME} complete at $(date '+%Y-%m-%d %H:%M:%S')"
