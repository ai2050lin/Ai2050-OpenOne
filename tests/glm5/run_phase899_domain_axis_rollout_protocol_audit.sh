#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ROUND_NAME="${1:-domain_axis_rollout_protocol_audit}"

COMMON_ARGS=(
  --round-name "${ROUND_NAME}"
  --phase898-round domain_axis_holdout_validation
  --max-sources-per-model 8
  --max-conditions-per-source 24
  --max-new-tokens 12
  --scale-up-factor 2.0
  --log-every 8
)

for MODEL in qwen3 glm4 deepseek7b; do
  python tests/glm5/phase899_domain_axis_rollout_protocol_audit.py \
    --model "${MODEL}" \
    "${COMMON_ARGS[@]}"
done

python tests/glm5/phase899_domain_axis_rollout_protocol_audit.py \
  --summarize-round \
  --round-name "${ROUND_NAME}"
