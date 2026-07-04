#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ROUND_NAME="${1:-route_preserving_blocker_band_disentanglement}"

COMMON_ARGS=(
  --round-name "${ROUND_NAME}"
  --phase899-round domain_axis_rollout_protocol_audit
  --max-rows-per-model 0
  --max-prefix-tokens 5
  --scale-up-factor 2.0
  --factors 0.75,0.5,0.25
  --band-size 32
  --span-kinds prompt_all,prompt_first8,prompt_last8,answer_prefix_all,last8_before_period,period_token
  --mlp-group-kinds band16_support_32,band16_support_64,band32_support_64,top_abs_64,low_abs_64
  --mlp-candidate-pool 512
  --log-every 4
)

for MODEL in qwen3 glm4 deepseek7b; do
  python tests/glm5/phase913_route_preserving_blocker_band_disentanglement.py \
    --model "${MODEL}" \
    "${COMMON_ARGS[@]}"
done

python tests/glm5/phase913_route_preserving_blocker_band_disentanglement.py \
  --summarize-round \
  --round-name "${ROUND_NAME}"
