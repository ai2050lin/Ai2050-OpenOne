#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-main}"
shift || true

MODELS=("qwen3" "glm4" "deepseek7b")
OUT_DIR="results/glm5_phase744_competitor_suppression_source_localization/${ROUND_NAME}"
mkdir -p "${OUT_DIR}"

COMMON_ARGS=("$@")

echo "[phase744] round=${ROUND_NAME} start at $(date '+%F %T')"

for model in "${MODELS[@]}"; do
  echo "[phase744] start model=${model} at $(date '+%F %T')"
  python tests/gpt5/phase744_competitor_suppression_source_localization.py \
    --model "${model}" \
    --round-name "${ROUND_NAME}" \
    --hard-exit-after-model \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "${OUT_DIR}/phase744_${model}.log"
  echo "[phase744] done model=${model} at $(date '+%F %T')"
done

python tests/gpt5/phase744_competitor_suppression_source_localization.py \
  --round-name "${ROUND_NAME}" \
  --summarize-only \
  "${COMMON_ARGS[@]}" \
  2>&1 | tee "${OUT_DIR}/phase744_summary.log"

echo "[phase744] round=${ROUND_NAME} complete at $(date '+%F %T')"
