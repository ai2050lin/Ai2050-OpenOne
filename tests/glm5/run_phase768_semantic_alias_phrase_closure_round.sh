#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-smoke}"
shift || true

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="results/glm5_phase768_semantic_alias_phrase_closure/${ROUND_NAME}"
RESULT_DIR="tests/result/phase768_semantic_alias_phrase_closure/${ROUND_NAME}"
mkdir -p "$OUT_DIR" "$RESULT_DIR"

MODELS=(qwen3 glm4 deepseek7b)

for MODEL_NAME in "${MODELS[@]}"; do
  echo "[phase768] $(date '+%Y-%m-%d %H:%M:%S') start ${MODEL_NAME} round=${ROUND_NAME}" | tee -a "${OUT_DIR}/phase768_summary.log"
  python tests/glm5/phase768_semantic_alias_phrase_closure.py \
    --model "${MODEL_NAME}" \
    --round-name "${ROUND_NAME}" \
    --hard-exit-after-model \
    "$@" 2>&1 | tee "${OUT_DIR}/phase768_${MODEL_NAME}.log"
  echo "[phase768] $(date '+%Y-%m-%d %H:%M:%S') done ${MODEL_NAME} round=${ROUND_NAME}" | tee -a "${OUT_DIR}/phase768_summary.log"
  sleep 3
done

python tests/glm5/phase768_semantic_alias_phrase_closure.py \
  --round-name "${ROUND_NAME}" \
  --write-cross-summary

cp "${OUT_DIR}/phase768_summary.log" "${RESULT_DIR}/phase768_summary.log"
echo "[phase768] $(date '+%Y-%m-%d %H:%M:%S') cross summary complete round=${ROUND_NAME}" | tee -a "${OUT_DIR}/phase768_summary.log"
