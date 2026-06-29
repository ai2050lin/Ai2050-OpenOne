#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${PHASE736_ROUND_NAME:-main}"
PHASE735_ROUND="${PHASE736_PHASE735_ROUND:-confirm}"
MAX_PAIRS="${PHASE736_MAX_PAIRS:-8}"
TOP_PATHS="${PHASE736_TOP_PATHS:-4}"
PREFERRED_SOURCES="${PHASE736_PREFERRED_SOURCES:-target_value_tokens,target_record_line,records_all,self_last,instruction,all_pre_answer}"
MAX_NEW_TOKENS="${PHASE736_MAX_NEW_TOKENS:-3}"
LOG_EVERY="${PHASE736_LOG_EVERY:-2}"
INCLUDE_EXTENDED="${PHASE736_INCLUDE_EXTENDED_RELATIONS:-0}"

OUT_DIR="results/glm5_phase736_source_replacement_generation_closure/${ROUND_NAME}"
mkdir -p "${OUT_DIR}"

echo "[phase736] round=${ROUND_NAME} phase735_round=${PHASE735_ROUND} pairs=${MAX_PAIRS} top_paths=${TOP_PATHS} max_new=${MAX_NEW_TOKENS}"

for MODEL in qwen3 glm4 deepseek7b; do
  echo "[phase736] start model=${MODEL} at $(date '+%Y-%m-%d %H:%M:%S')"
  CMD=(
    python tests/gpt5/phase736_source_replacement_generation_closure.py
    --model "${MODEL}"
    --round-name "${ROUND_NAME}"
    --phase735-round "${PHASE735_ROUND}"
    --max-pairs "${MAX_PAIRS}"
    --top-paths "${TOP_PATHS}"
    --preferred-sources "${PREFERRED_SOURCES}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --log-every "${LOG_EVERY}"
    --hard-exit-after-model
  )
  if [[ "${INCLUDE_EXTENDED}" == "1" ]]; then
    CMD+=(--include-extended-relations)
  fi
  "${CMD[@]}" 2>&1 | tee "${OUT_DIR}/phase736_${MODEL}.log"
  echo "[phase736] done model=${MODEL} at $(date '+%Y-%m-%d %H:%M:%S')"
done

python tests/gpt5/phase736_source_replacement_generation_closure.py \
  --round-name "${ROUND_NAME}" \
  --summarize-only 2>&1 | tee "${OUT_DIR}/phase736_summary.log"

echo "[phase736] round=${ROUND_NAME} complete at $(date '+%Y-%m-%d %H:%M:%S')"
