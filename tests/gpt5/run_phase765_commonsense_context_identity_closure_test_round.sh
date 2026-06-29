#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-smoke}"
shift || true

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="results/glm5_phase765_commonsense_context_identity_closure_test/${ROUND_NAME}"
mkdir -p "$OUT_DIR"

MODELS=(qwen3 glm4 deepseek7b)

for MODEL_NAME in "${MODELS[@]}"; do
  echo "[phase765] $(date '+%Y-%m-%d %H:%M:%S') start ${MODEL_NAME} round=${ROUND_NAME}" | tee -a "${OUT_DIR}/phase765_summary.log"
  python tests/gpt5/phase765_commonsense_context_identity_closure_test.py \
    --model "${MODEL_NAME}" \
    --round-name "${ROUND_NAME}" \
    --hard-exit-after-model \
    "$@" 2>&1 | tee "${OUT_DIR}/phase765_${MODEL_NAME}.log"
  echo "[phase765] $(date '+%Y-%m-%d %H:%M:%S') done ${MODEL_NAME} round=${ROUND_NAME}" | tee -a "${OUT_DIR}/phase765_summary.log"
  sleep 3
done

python tests/gpt5/phase765_commonsense_context_identity_closure_test.py \
  --round-name "${ROUND_NAME}" \
  --write-cross-summary

echo "[phase765] $(date '+%Y-%m-%d %H:%M:%S') cross summary complete round=${ROUND_NAME}" | tee -a "${OUT_DIR}/phase765_summary.log"
