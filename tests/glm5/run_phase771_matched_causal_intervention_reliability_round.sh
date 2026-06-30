#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-smoke}"
shift || true

OUT_DIR="results/glm5_phase771_matched_causal_intervention_reliability_test/${ROUND_NAME}"
RESULT_DIR="tests/result/phase771_matched_causal_intervention_reliability_test/${ROUND_NAME}"
mkdir -p "${OUT_DIR}" "${RESULT_DIR}"

MODELS=(qwen3 glm4 deepseek7b)

for MODEL_NAME in "${MODELS[@]}"; do
  echo "[phase771] $(date '+%Y-%m-%d %H:%M:%S') start ${MODEL_NAME} round=${ROUND_NAME}" | tee -a "${OUT_DIR}/phase771_summary.log"
  python tests/glm5/phase771_matched_causal_intervention_reliability_test.py \
    --model "${MODEL_NAME}" \
    --round-name "${ROUND_NAME}" \
    --hard-exit-after-model \
    "$@" 2>&1 | tee "${OUT_DIR}/phase771_${MODEL_NAME}.log"
  echo "[phase771] $(date '+%Y-%m-%d %H:%M:%S') done ${MODEL_NAME} round=${ROUND_NAME}" | tee -a "${OUT_DIR}/phase771_summary.log"
done

python tests/glm5/phase771_matched_causal_intervention_reliability_test.py \
  --round-name "${ROUND_NAME}" \
  --summarize-only

cp "${OUT_DIR}/phase771_summary.log" "${RESULT_DIR}/phase771_summary.log"
echo "[phase771] $(date '+%Y-%m-%d %H:%M:%S') cross summary complete round=${ROUND_NAME}" | tee -a "${OUT_DIR}/phase771_summary.log"
