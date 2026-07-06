#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-stop_execution_source_localization_audit}"
SCRIPT="tests/gpt5/phase205_stop_execution_source_localization_audit.py"

run_model() {
  local model="$1"
  echo "== Phase205 ${model} =="
  python "${SCRIPT}" \
    --model "${model}" \
    --round-name "${ROUND_NAME}" \
    --phase204-round global_trajectory_stop_execution_atlas \
    --max-trajectories 36 \
    --top-channels-per-layer 5 \
    --batch-size 8
}

run_model qwen3
run_model glm4
run_model deepseek7b

python "${SCRIPT}" --round-name "${ROUND_NAME}" --summarize-round
