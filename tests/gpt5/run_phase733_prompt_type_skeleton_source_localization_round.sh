#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${PHASE733_ROUND_NAME:-main}"
MAX_SCAN_CASES="${PHASE733_MAX_SCAN_CASES:-36}"
MAX_PAIRS="${PHASE733_MAX_PAIRS:-12}"
MAX_CANDIDATE_SITES="${PHASE733_MAX_CANDIDATE_SITES:-6}"
MAX_NEW_TOKENS="${PHASE733_MAX_NEW_TOKENS:-4}"
LOG_EVERY="${PHASE733_LOG_EVERY:-4}"

OUT_DIR="results/glm5_phase733_prompt_type_skeleton_source_localization/${ROUND_NAME}"
mkdir -p "${OUT_DIR}"

echo "[phase733] round=${ROUND_NAME} scan_cases=${MAX_SCAN_CASES} pairs=${MAX_PAIRS} candidates=${MAX_CANDIDATE_SITES}"

for MODEL in qwen3 glm4 deepseek7b; do
  echo "[phase733] start model=${MODEL} at $(date '+%Y-%m-%d %H:%M:%S')"
  python tests/gpt5/phase733_prompt_type_skeleton_source_localization.py \
    --model "${MODEL}" \
    --round-name "${ROUND_NAME}" \
    --max-scan-cases "${MAX_SCAN_CASES}" \
    --max-pairs "${MAX_PAIRS}" \
    --max-candidate-sites "${MAX_CANDIDATE_SITES}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --log-every "${LOG_EVERY}" \
    --hard-exit-after-model 2>&1 | tee "${OUT_DIR}/phase733_${MODEL}.log"
  echo "[phase733] done model=${MODEL} at $(date '+%Y-%m-%d %H:%M:%S')"
done

python tests/gpt5/phase733_prompt_type_skeleton_source_localization.py \
  --round-name "${ROUND_NAME}" \
  --summarize-only 2>&1 | tee "${OUT_DIR}/phase733_summary.log"

echo "[phase733] round=${ROUND_NAME} complete at $(date '+%Y-%m-%d %H:%M:%S')"
