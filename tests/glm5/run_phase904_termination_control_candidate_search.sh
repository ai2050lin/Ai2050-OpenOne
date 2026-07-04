#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ROUND_NAME="${1:-termination_control_candidate_search}"

COMMON_ARGS=(
  --round-name "${ROUND_NAME}"
  --phase899-round domain_axis_rollout_protocol_audit
  --phase903-round protocol_continuation_field_mapping
  --max-rows-per-model 0
  --max-candidates 8
  --max-prefix-tokens 5
  --max-suffix-tokens 8
  --suppress-steps 2
  --scale-up-factor 2.0
  --log-every 8
)

for MODEL in qwen3 glm4 deepseek7b; do
  python tests/glm5/phase904_termination_control_candidate_search.py \
    --model "${MODEL}" \
    "${COMMON_ARGS[@]}"
done

python tests/glm5/phase904_termination_control_candidate_search.py \
  --summarize-round \
  --round-name "${ROUND_NAME}"
