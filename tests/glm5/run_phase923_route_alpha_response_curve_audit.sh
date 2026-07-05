#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ROUND_NAME="${1:-route_alpha_response_curve_audit}"

COMMON_ARGS=(
  --round-name "${ROUND_NAME}"
  --phase915-round near_boundary_action_gate_search
  --source-control-label L39_mlp_output_scale_1.5
  --boundary-blocker-token a
  --max-candidates-per-model 12
  --target-layer 39
  --max-prefix-tokens 5
  --scale-up-factor 2.0
  --l4-candidate-pool 512
  --channel-candidate-pool 768
  --band-size 32
  --group-budget 64
  --low-factors 1.125,1.25,1.375
  --route-alphas 0.5,0.625,0.75,0.875,1.0,1.125,1.25,1.375,1.5
  --log-every 2
)

for MODEL in qwen3 glm4 deepseek7b; do
  python tests/glm5/phase923_route_alpha_response_curve_audit.py \
    --model "${MODEL}" \
    "${COMMON_ARGS[@]}"
done

python tests/glm5/phase923_route_alpha_response_curve_audit.py \
  --summarize-round \
  --round-name "${ROUND_NAME}"
