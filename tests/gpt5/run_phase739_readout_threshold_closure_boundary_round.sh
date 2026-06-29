#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${PHASE739_ROUND_NAME:-main}"
MAX_PAIRS="${PHASE739_MAX_PAIRS:-8}"
TOP_AUDITS="${PHASE739_TOP_AUDITS:-2}"
ALPHA_MAX="${PHASE739_ALPHA_MAX:-80}"
MAX_NEW_TOKENS="${PHASE739_MAX_NEW_TOKENS:-4}"
LOG_EVERY="${PHASE739_LOG_EVERY:-2}"
PHASE738_ROUND="${PHASE739_PHASE738_ROUND:-confirm}"
NO_GENERATION="${PHASE739_NO_GENERATION:-0}"
OUT_DIR="results/glm5_phase739_readout_threshold_closure_boundary/${ROUND_NAME}"
mkdir -p "${OUT_DIR}"

COMMON_ARGS=(
  --round-name "${ROUND_NAME}"
  --phase738-round "${PHASE738_ROUND}"
  --max-pairs "${MAX_PAIRS}"
  --top-audits "${TOP_AUDITS}"
  --alpha-max "${ALPHA_MAX}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --log-every "${LOG_EVERY}"
)

if [[ "${NO_GENERATION}" == "1" ]]; then
  COMMON_ARGS+=(--no-generation)
fi

for model in qwen3 glm4 deepseek7b; do
  echo "[phase739] start model=${model} at $(date '+%F %T')"
  python tests/gpt5/phase739_readout_threshold_closure_boundary.py \
    --model "${model}" \
    "${COMMON_ARGS[@]}" \
    --hard-exit-after-model \
    2>&1 | tee "${OUT_DIR}/phase739_${model}.log"
  echo "[phase739] done model=${model} at $(date '+%F %T')"
done

python tests/gpt5/phase739_readout_threshold_closure_boundary.py \
  --round-name "${ROUND_NAME}" \
  --summarize-only \
  2>&1 | tee "${OUT_DIR}/phase739_summary.log"

echo "[phase739] round=${ROUND_NAME} complete at $(date '+%F %T')"
