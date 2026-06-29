#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${PHASE738_ROUND_NAME:-main}"
MAX_PAIRS="${PHASE738_MAX_PAIRS:-8}"
TOP_AUDITS="${PHASE738_TOP_AUDITS:-5}"
PHASE737_ROUND="${PHASE738_PHASE737_ROUND:-confirm}"
LOG_EVERY="${PHASE738_LOG_EVERY:-2}"

echo "[phase738] round=${ROUND_NAME} phase737_round=${PHASE737_ROUND} pairs=${MAX_PAIRS} top_audits=${TOP_AUDITS}"
mkdir -p "results/glm5_phase738_readout_margin_continuation_audit/${ROUND_NAME}"

for MODEL in qwen3 glm4 deepseek7b; do
  echo "[phase738] start model=${MODEL} at $(date '+%Y-%m-%d %H:%M:%S')"
  python tests/gpt5/phase738_readout_margin_continuation_audit.py \
    --model "${MODEL}" \
    --round-name "${ROUND_NAME}" \
    --phase737-round "${PHASE737_ROUND}" \
    --max-pairs "${MAX_PAIRS}" \
    --top-audits "${TOP_AUDITS}" \
    --log-every "${LOG_EVERY}" \
    --hard-exit-after-model \
    2>&1 | tee "results/glm5_phase738_readout_margin_continuation_audit/${ROUND_NAME}/phase738_${MODEL}.log"
  echo "[phase738] done model=${MODEL} at $(date '+%Y-%m-%d %H:%M:%S')"
done

python tests/gpt5/phase738_readout_margin_continuation_audit.py --round-name "${ROUND_NAME}" --summarize-only
echo "[phase738] round=${ROUND_NAME} complete at $(date '+%Y-%m-%d %H:%M:%S')"
