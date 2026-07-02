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
    ;;
  *)
    SPLIT_TYPES="in_sample,object_holdout,prompt_holdout"
    ;;
esac

echo "[Phase847] round=${ROUND_NAME} source_phase845=${PHASE845_ROUND} split_types=${SPLIT_TYPES}"

for MODEL in qwen3 glm4 deepseek7b; do
  echo "[Phase847] running ${MODEL}"
  python tests/glm5/phase847_context_gated_residual_audit.py \
    --model "${MODEL}" \
    --round-name "${ROUND_NAME}" \
    --phase845-round "${PHASE845_ROUND}" \
    --split-types "${SPLIT_TYPES}" \
    --log-every 1
done

python tests/glm5/phase847_context_gated_residual_audit.py \
  --round-name "${ROUND_NAME}" \
  --summarize-only

