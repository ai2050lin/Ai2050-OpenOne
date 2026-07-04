#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ROUND_NAME="${1:-near_boundary_action_gate_search}"

COMMON_ARGS=(
  --round-name "${ROUND_NAME}"
  --phase914-round l4_mlp_route_near_holdout_validation
  --max-candidates-per-model 12
  --boundary-factors 0.3,0.4
  --max-prefix-tokens 5
  --scale-up-factor 2.0
  --mlp-candidate-pool 512
  --band-size 32
  --action-sites l0_output,L-1:mlp,L-1:attn,L-4:mlp,L-4:attn
  --direction-kinds eos_minus_blocker_top1,minus_blocker_top1,minus_blocker_top3_mean,eos_boost
  --betas 0.05,0.1,0.25,0.5
  --component-scales 0.0,0.5,1.5
  --log-every 2
)

for MODEL in qwen3 glm4 deepseek7b; do
  python tests/glm5/phase915_near_boundary_action_gate_search.py \
    --model "${MODEL}" \
    "${COMMON_ARGS[@]}"
done

python tests/glm5/phase915_near_boundary_action_gate_search.py \
  --summarize-round \
  --round-name "${ROUND_NAME}"
