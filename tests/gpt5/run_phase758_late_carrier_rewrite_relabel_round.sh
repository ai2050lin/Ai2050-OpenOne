#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-main}"
shift || true

MODELS=("qwen3" "glm4" "deepseek7b")
OUT_DIR="results/glm5_phase758_late_carrier_rewrite_relabel/${ROUND_NAME}"
mkdir -p "${OUT_DIR}"

COMMON_ARGS=("$@")

echo "[phase758] round=${ROUND_NAME} start at $(date '+%F %T')"

for model in "${MODELS[@]}"; do
  echo "[phase758] start model=${model} at $(date '+%F %T')"
  python tests/gpt5/phase758_late_carrier_rewrite_relabel.py \
    --model "${model}" \
    --round-name "${ROUND_NAME}" \
    --hard-exit-after-model \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "${OUT_DIR}/phase758_${model}.log"
  echo "[phase758] done model=${model} at $(date '+%F %T')"
done

python tests/gpt5/phase758_late_carrier_rewrite_relabel.py \
  --round-name "${ROUND_NAME}" \
  --summarize-only \
  "${COMMON_ARGS[@]}" \
  2>&1 | tee "${OUT_DIR}/phase758_summary.log"

echo "[phase758] round=${ROUND_NAME} complete at $(date '+%F %T')"
