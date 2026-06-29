#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${PHASE740_ROUND_NAME:-main}"
MAX_PAIRS="${PHASE740_MAX_PAIRS:-8}"
TOP_AUDITS="${PHASE740_TOP_AUDITS:-1}"
SCAN_LAST_N="${PHASE740_SCAN_LAST_N:-0}"
LOG_EVERY="${PHASE740_LOG_EVERY:-2}"
PHASE739_ROUND="${PHASE740_PHASE739_ROUND:-confirm}"
PHASE738_ROUND="${PHASE740_PHASE738_ROUND:-confirm}"
OUT_DIR="results/glm5_phase740_natural_readout_boost_source_backtrace/${ROUND_NAME}"
mkdir -p "${OUT_DIR}"

COMMON_ARGS=(
  --round-name "${ROUND_NAME}"
  --phase739-round "${PHASE739_ROUND}"
  --phase738-round "${PHASE738_ROUND}"
  --max-pairs "${MAX_PAIRS}"
  --top-audits "${TOP_AUDITS}"
  --scan-last-n "${SCAN_LAST_N}"
  --log-every "${LOG_EVERY}"
)

for model in qwen3 glm4 deepseek7b; do
  echo "[phase740] start model=${model} at $(date '+%F %T')"
  python tests/gpt5/phase740_natural_readout_boost_source_backtrace.py \
    --model "${model}" \
    "${COMMON_ARGS[@]}" \
    --hard-exit-after-model \
    2>&1 | tee "${OUT_DIR}/phase740_${model}.log"
  echo "[phase740] done model=${model} at $(date '+%F %T')"
done

python tests/gpt5/phase740_natural_readout_boost_source_backtrace.py \
  --round-name "${ROUND_NAME}" \
  --summarize-only \
  2>&1 | tee "${OUT_DIR}/phase740_summary.log"

echo "[phase740] round=${ROUND_NAME} complete at $(date '+%F %T')"
