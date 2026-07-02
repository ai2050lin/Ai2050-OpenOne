#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-smoke}"
PHASE845_ROUND="${2:-$ROUND_NAME}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

case "${ROUND_NAME}" in
  smoke)
    SPLIT_TYPES="in_sample"
    LOG_EVERY=1
    ;;
  main)
    SPLIT_TYPES="in_sample,object_holdout,prompt_holdout"
    LOG_EVERY=1
    ;;
  confirm)
    SPLIT_TYPES="in_sample,object_holdout,prompt_holdout"
    LOG_EVERY=1
    ;;
  *)
    SPLIT_TYPES="in_sample,object_holdout,prompt_holdout"
    LOG_EVERY=1
    ;;
esac

echo "[Phase846] round=${ROUND_NAME} source_phase845=${PHASE845_ROUND} split_types=${SPLIT_TYPES}"

for MODEL in qwen3 glm4 deepseek7b; do
  echo "[Phase846] running ${MODEL}"
  python tests/glm5/phase846_geometry_boundary_equation_fitting.py \
    --model "${MODEL}" \
    --round-name "${ROUND_NAME}" \
    --phase845-round "${PHASE845_ROUND}" \
    --split-types "${SPLIT_TYPES}" \
    --log-every "${LOG_EVERY}"
done

python tests/glm5/phase846_geometry_boundary_equation_fitting.py \
  --round-name "${ROUND_NAME}" \
  --summarize-only

