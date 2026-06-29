#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${PHASE742_ROUND_NAME:-main}"
MAX_PAIRS="${PHASE742_MAX_PAIRS:-6}"
TOP_AUDITS="${PHASE742_TOP_AUDITS:-2}"
TOP_CANDIDATES="${PHASE742_TOP_CANDIDATES:-3}"
LOG_EVERY="${PHASE742_LOG_EVERY:-2}"
PHASE739_ROUND="${PHASE742_PHASE739_ROUND:-confirm}"
PHASE741_ROUND="${PHASE742_PHASE741_ROUND:-confirm}"
OUT_DIR="results/glm5_phase742_combined_threshold_component_closure/${ROUND_NAME}"
mkdir -p "${OUT_DIR}"

COMMON_ARGS=(
  --round-name "${ROUND_NAME}"
  --phase739-round "${PHASE739_ROUND}"
  --phase741-round "${PHASE741_ROUND}"
  --max-pairs "${MAX_PAIRS}"
  --top-audits "${TOP_AUDITS}"
  --top-candidates "${TOP_CANDIDATES}"
  --log-every "${LOG_EVERY}"
)

for model in qwen3 glm4 deepseek7b; do
  echo "[phase742] start model=${model} at $(date '+%F %T')"
  python tests/gpt5/phase742_combined_threshold_component_closure.py \
    --model "${model}" \
    "${COMMON_ARGS[@]}" \
    --hard-exit-after-model \
    2>&1 | tee "${OUT_DIR}/phase742_${model}.log"
  echo "[phase742] done model=${model} at $(date '+%F %T')"
done

python tests/gpt5/phase742_combined_threshold_component_closure.py \
  --round-name "${ROUND_NAME}" \
  --summarize-only \
  2>&1 | tee "${OUT_DIR}/phase742_summary.log"

echo "[phase742] round=${ROUND_NAME} complete at $(date '+%F %T')"
