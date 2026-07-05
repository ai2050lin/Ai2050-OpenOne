#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ROUND_NAME="${1:-l39_mlp_channel_a_blocker_suppressor_localization}"

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
  --up-groups eos_support_32,eos_support_64,margin_support_pos_32,margin_support_pos_64
  --down-groups a_blocker_support_32,a_blocker_support_64,a_logit_support_64,margin_support_neg_32,margin_support_neg_64,band_blocker_support_64
  --general-groups top_abs_64,low_abs_64
  --up-factors 1.25,1.5,2.0
  --down-factors 0.0,0.25,0.5,0.75
  --general-factors 0.0,0.5,1.5,2.0
  --log-every 2
)

for MODEL in qwen3 glm4 deepseek7b; do
  python tests/glm5/phase918_l39_mlp_channel_a_blocker_suppressor_localization.py \
    --model "${MODEL}" \
    "${COMMON_ARGS[@]}"
done

python tests/glm5/phase918_l39_mlp_channel_a_blocker_suppressor_localization.py \
  --summarize-round \
  --round-name "${ROUND_NAME}"
