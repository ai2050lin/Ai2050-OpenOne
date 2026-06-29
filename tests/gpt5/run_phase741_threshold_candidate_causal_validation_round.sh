#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${PHASE741_ROUND_NAME:-main}"
MAX_PAIRS="${PHASE741_MAX_PAIRS:-6}"
TOP_AUDITS="${PHASE741_TOP_AUDITS:-2}"
TOP_CANDIDATES="${PHASE741_TOP_CANDIDATES:-3}"
LOG_EVERY="${PHASE741_LOG_EVERY:-2}"
PHASE739_ROUND="${PHASE741_PHASE739_ROUND:-confirm}"
PHASE740_ROUND="${PHASE741_PHASE740_ROUND:-confirm}"
OUT_DIR="results/glm5_phase741_threshold_candidate_causal_validation/${ROUND_NAME}"
mkdir -p "${OUT_DIR}"

COMMON_ARGS=(
  --round-name "${ROUND_NAME}"
  --phase739-round "${PHASE739_ROUND}"
  --phase740-round "${PHASE740_ROUND}"
  --max-pairs "${MAX_PAIRS}"
  --top-audits "${TOP_AUDITS}"
  --top-candidates "${TOP_CANDIDATES}"
  --log-every "${LOG_EVERY}"
)

for model in qwen3 glm4 deepseek7b; do
  echo "[phase741] start model=${model} at $(date '+%F %T')"
  python tests/gpt5/phase741_threshold_candidate_causal_validation.py \
    --model "${model}" \
    "${COMMON_ARGS[@]}" \
    --hard-exit-after-model \
    2>&1 | tee "${OUT_DIR}/phase741_${model}.log"
  echo "[phase741] done model=${model} at $(date '+%F %T')"
done

python tests/gpt5/phase741_threshold_candidate_causal_validation.py \
  --round-name "${ROUND_NAME}" \
  --summarize-only \
  2>&1 | tee "${OUT_DIR}/phase741_summary.log"

echo "[phase741] round=${ROUND_NAME} complete at $(date '+%F %T')"
