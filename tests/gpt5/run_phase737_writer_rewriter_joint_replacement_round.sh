#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${PHASE737_ROUND_NAME:-main}"
MAX_PAIRS="${PHASE737_MAX_PAIRS:-4}"
TOP_PATHS="${PHASE737_TOP_PATHS:-3}"
TOP_MLP="${PHASE737_TOP_MLP:-2}"
MAX_NEW_TOKENS="${PHASE737_MAX_NEW_TOKENS:-3}"
MODE_SET="${PHASE737_MODE_SET:-compact}"
LOG_EVERY="${PHASE737_LOG_EVERY:-1}"
PREFERRED_SOURCES="${PHASE737_PREFERRED_SOURCES:-}"
PHASE735_ROUND="${PHASE737_PHASE735_ROUND:-confirm}"

echo "[phase737] round=${ROUND_NAME} phase735_round=${PHASE735_ROUND} pairs=${MAX_PAIRS} top_paths=${TOP_PATHS} top_mlp=${TOP_MLP} mode=${MODE_SET} max_new=${MAX_NEW_TOKENS}"
mkdir -p "results/glm5_phase737_writer_rewriter_joint_replacement/${ROUND_NAME}"

for MODEL in qwen3 glm4 deepseek7b; do
  echo "[phase737] start model=${MODEL} at $(date '+%Y-%m-%d %H:%M:%S')"
  CMD=(
    python tests/gpt5/phase737_writer_rewriter_joint_replacement.py
    --model "${MODEL}"
    --round-name "${ROUND_NAME}"
    --phase735-round "${PHASE735_ROUND}"
    --max-pairs "${MAX_PAIRS}"
    --top-paths "${TOP_PATHS}"
    --top-mlp "${TOP_MLP}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --mode-set "${MODE_SET}"
    --log-every "${LOG_EVERY}"
    --hard-exit-after-model
  )
  if [[ -n "${PREFERRED_SOURCES}" ]]; then
    CMD+=(--preferred-sources "${PREFERRED_SOURCES}")
  fi
  "${CMD[@]}" 2>&1 | tee "results/glm5_phase737_writer_rewriter_joint_replacement/${ROUND_NAME}/phase737_${MODEL}.log"
  echo "[phase737] done model=${MODEL} at $(date '+%Y-%m-%d %H:%M:%S')"
done

python tests/gpt5/phase737_writer_rewriter_joint_replacement.py --round-name "${ROUND_NAME}" --summarize-only
echo "[phase737] round=${ROUND_NAME} complete at $(date '+%Y-%m-%d %H:%M:%S')"
