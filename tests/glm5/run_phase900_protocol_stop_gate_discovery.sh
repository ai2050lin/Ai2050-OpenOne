#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ROUND_NAME="${1:-protocol_stop_gate_discovery}"

COMMON_ARGS=(
  --round-name "${ROUND_NAME}"
  --phase899-round domain_axis_rollout_protocol_audit
  --max-rows-per-model 0
  --max-new-tokens 12
  --scale-up-factor 2.0
  --max-same-layer-heads 4
  --log-every 8
)

for MODEL in qwen3 glm4 deepseek7b; do
  python tests/glm5/phase900_protocol_stop_gate_discovery.py \
    --model "${MODEL}" \
    "${COMMON_ARGS[@]}"
done

python tests/glm5/phase900_protocol_stop_gate_discovery.py \
  --summarize-round \
  --round-name "${ROUND_NAME}"
