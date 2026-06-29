#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-main}"
shift || true

MODELS=("qwen3" "glm4" "deepseek7b")
OUT_DIR="results/glm5_phase748_natural_route_suppressor_matrix/${ROUND_NAME}"
mkdir -p "${OUT_DIR}"

COMMON_ARGS=("$@")

echo "[phase748] round=${ROUND_NAME} start at $(date '+%F %T')"

for model in "${MODELS[@]}"; do
  echo "[phase748] start model=${model} at $(date '+%F %T')"
  python tests/gpt5/phase748_natural_route_suppressor_matrix.py \
    --model "${model}" \
    --round-name "${ROUND_NAME}" \
    --hard-exit-after-model \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "${OUT_DIR}/phase748_${model}.log"
  echo "[phase748] done model=${model} at $(date '+%F %T')"
done

python tests/gpt5/phase748_natural_route_suppressor_matrix.py \
  --round-name "${ROUND_NAME}" \
  --summarize-only \
  "${COMMON_ARGS[@]}" \
  2>&1 | tee "${OUT_DIR}/phase748_summary.log"

echo "[phase748] round=${ROUND_NAME} complete at $(date '+%F %T')"
