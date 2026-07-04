#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ROUND_NAME="${1:-stop_token_competitiveness_audit}"

COMMON_ARGS=(
  --round-name "${ROUND_NAME}"
  --phase899-round domain_axis_rollout_protocol_audit
  --max-rows-per-model 0
  --max-prefix-tokens 5
  --scale-up-factor 2.0
  --log-every 16
)

for MODEL in qwen3 glm4 deepseek7b; do
  python tests/glm5/phase901_stop_token_competitiveness_audit.py \
    --model "${MODEL}" \
    "${COMMON_ARGS[@]}"
done

python tests/glm5/phase901_stop_token_competitiveness_audit.py \
  --summarize-round \
  --round-name "${ROUND_NAME}"
