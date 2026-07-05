#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ROUND_NAME="${1:-frozen_l39_signed_margin_group_transfer_validation}"

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
  --up-groups margin_support_pos_64,eos_support_64
  --down-groups a_blocker_support_64,margin_support_neg_64,a_logit_support_64
  --margin-pos-factors 1.375,1.5,1.75,2.0
  --eos-factors 1.75,2.0
  --down-factors 0.0,0.125,0.25,0.375,0.5
  --include-self-transfer
  --log-every 2
)

for MODEL in qwen3 glm4 deepseek7b; do
  python tests/glm5/phase919_frozen_l39_signed_margin_group_transfer_validation.py \
    --model "${MODEL}" \
    "${COMMON_ARGS[@]}"
done

python tests/glm5/phase919_frozen_l39_signed_margin_group_transfer_validation.py \
  --summarize-round \
  --round-name "${ROUND_NAME}"
