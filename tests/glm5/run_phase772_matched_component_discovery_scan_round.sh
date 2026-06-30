#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-smoke}"
shift || true

OUT_DIR="results/glm5_phase772_matched_component_discovery_scan/${ROUND_NAME}"
RESULT_DIR="tests/result/phase772_matched_component_discovery_scan/${ROUND_NAME}"
mkdir -p "${OUT_DIR}" "${RESULT_DIR}"

MODELS=(qwen3 glm4 deepseek7b)

for MODEL_NAME in "${MODELS[@]}"; do
  echo "[phase772] $(date '+%Y-%m-%d %H:%M:%S') start ${MODEL_NAME} round=${ROUND_NAME}" | tee -a "${OUT_DIR}/phase772_summary.log"
  python tests/glm5/phase772_matched_component_discovery_scan.py \
    --model "${MODEL_NAME}" \
    --round-name "${ROUND_NAME}" \
    --hard-exit-after-model \
    "$@" 2>&1 | tee "${OUT_DIR}/phase772_${MODEL_NAME}.log"
  echo "[phase772] $(date '+%Y-%m-%d %H:%M:%S') done ${MODEL_NAME} round=${ROUND_NAME}" | tee -a "${OUT_DIR}/phase772_summary.log"
done

python tests/glm5/phase772_matched_component_discovery_scan.py \
  --round-name "${ROUND_NAME}" \
  --summarize-only

cp "${OUT_DIR}/phase772_summary.log" "${RESULT_DIR}/phase772_summary.log"
echo "[phase772] $(date '+%Y-%m-%d %H:%M:%S') cross summary complete round=${ROUND_NAME}" | tee -a "${OUT_DIR}/phase772_summary.log"
