#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-smoke}"
shift || true

OUT_DIR="results/glm5_phase773_instruction_source_disentanglement/${ROUND_NAME}"
RESULT_DIR="tests/result/phase773_instruction_source_disentanglement/${ROUND_NAME}"
mkdir -p "${OUT_DIR}" "${RESULT_DIR}"

MODELS=(qwen3 glm4 deepseek7b)

for MODEL_NAME in "${MODELS[@]}"; do
  echo "[phase773] $(date '+%Y-%m-%d %H:%M:%S') start ${MODEL_NAME} round=${ROUND_NAME}" | tee -a "${OUT_DIR}/phase773_summary.log"
  python tests/glm5/phase773_instruction_source_disentanglement.py \
    --model "${MODEL_NAME}" \
    --round-name "${ROUND_NAME}" \
    --hard-exit-after-model \
    "$@" 2>&1 | tee "${OUT_DIR}/phase773_${MODEL_NAME}.log"
  echo "[phase773] $(date '+%Y-%m-%d %H:%M:%S') done ${MODEL_NAME} round=${ROUND_NAME}" | tee -a "${OUT_DIR}/phase773_summary.log"
done

python tests/glm5/phase773_instruction_source_disentanglement.py \
  --round-name "${ROUND_NAME}" \
  --summarize-only

cp "${OUT_DIR}/phase773_summary.log" "${RESULT_DIR}/phase773_summary.log"
echo "[phase773] $(date '+%Y-%m-%d %H:%M:%S') cross summary complete round=${ROUND_NAME}" | tee -a "${OUT_DIR}/phase773_summary.log"
