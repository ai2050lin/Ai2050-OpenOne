#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-first_internal_trace_batch}"
MAX_TRACE_ROWS="${MAX_TRACE_ROWS:-100}"
MAX_ROLLOUT_STEPS="${MAX_ROLLOUT_STEPS:-4}"

for MODEL in qwen3 glm4 deepseek7b; do
  python tests/gpt5/phase244_first_internal_trace_batch.py \
    --model "${MODEL}" \
    --round-name "${ROUND_NAME}" \
    --max-trace-rows "${MAX_TRACE_ROWS}" \
    --max-rollout-steps "${MAX_ROLLOUT_STEPS}"
done

python tests/gpt5/phase244_first_internal_trace_batch.py \
  --round-name "${ROUND_NAME}" \
  --summarize
