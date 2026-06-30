#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-smoke}"
shift || true

OUT_DIR="results/glm5_phase778_surface_form_normalization_causal_audit/${ROUND_NAME}"
RESULT_DIR="tests/result/phase778_surface_form_normalization_causal_audit/${ROUND_NAME}"
mkdir -p "${OUT_DIR}" "${RESULT_DIR}"

MODELS=(qwen3 glm4 deepseek7b)

for MODEL_NAME in "${MODELS[@]}"; do
  echo "[phase778] $(date '+%Y-%m-%d %H:%M:%S') start ${MODEL_NAME} round=${ROUND_NAME}" | tee -a "${OUT_DIR}/phase778_summary.log"
  python tests/glm5/phase778_surface_form_normalization_causal_audit.py \
    --model "${MODEL_NAME}" \
    --round-name "${ROUND_NAME}" \
    --hard-exit-after-model \
    "$@" 2>&1 | tee "${OUT_DIR}/phase778_${MODEL_NAME}.log"
  echo "[phase778] $(date '+%Y-%m-%d %H:%M:%S') done ${MODEL_NAME} round=${ROUND_NAME}" | tee -a "${OUT_DIR}/phase778_summary.log"
done

python tests/glm5/phase778_surface_form_normalization_causal_audit.py \
  --round-name "${ROUND_NAME}" \
  --summarize-only

cp "${OUT_DIR}/phase778_summary.log" "${RESULT_DIR}/phase778_summary.log"
echo "[phase778] $(date '+%Y-%m-%d %H:%M:%S') cross summary complete round=${ROUND_NAME}" | tee -a "${OUT_DIR}/phase778_summary.log"
