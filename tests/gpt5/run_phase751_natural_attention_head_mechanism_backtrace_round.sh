#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-main}"
shift || true

MODELS=("qwen3" "glm4" "deepseek7b")
OUT_DIR="results/glm5_phase751_natural_attention_head_mechanism_backtrace/${ROUND_NAME}"
mkdir -p "${OUT_DIR}"

COMMON_ARGS=("$@")

echo "[phase751] round=${ROUND_NAME} start at $(date '+%F %T')"

for model in "${MODELS[@]}"; do
  echo "[phase751] start model=${model} at $(date '+%F %T')"
  python tests/gpt5/phase751_natural_attention_head_mechanism_backtrace.py \
    --model "${model}" \
    --round-name "${ROUND_NAME}" \
    --hard-exit-after-model \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "${OUT_DIR}/phase751_${model}.log"
  echo "[phase751] done model=${model} at $(date '+%F %T')"
done

python tests/gpt5/phase751_natural_attention_head_mechanism_backtrace.py \
  --round-name "${ROUND_NAME}" \
  --summarize-only \
  "${COMMON_ARGS[@]}" \
  2>&1 | tee "${OUT_DIR}/phase751_summary.log"

echo "[phase751] round=${ROUND_NAME} complete at $(date '+%F %T')"
