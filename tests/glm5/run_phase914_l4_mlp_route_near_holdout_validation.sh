#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ROUND_NAME="${1:-l4_mlp_route_near_holdout_validation}"

COMMON_ARGS=(
  --round-name "${ROUND_NAME}"
  --phase899-round domain_axis_rollout_protocol_audit
  --max-rows-per-model 0
  --max-eval-items-per-model 96
  --prompt-variants natural_question,natural_category,classification,question_plain,type_of_completion
  --holdout-prompt-variants natural_question,classification
  --same-domain-holdout-per-domain 4
  --max-prefix-tokens 5
  --scale-up-factor 2.0
  --factors 0.9,0.8,0.7,0.6,0.5,0.4,0.3
  --mlp-group-kinds top_abs_64,band16_support_32,band16_support_64,band32_support_64,low_abs_64
  --mlp-candidate-pool 512
  --band-size 32
  --route-topk-filter 50
  --log-every 8
)

for MODEL in qwen3 glm4 deepseek7b; do
  python tests/glm5/phase914_l4_mlp_route_near_holdout_validation.py \
    --model "${MODEL}" \
    "${COMMON_ARGS[@]}"
done

python tests/glm5/phase914_l4_mlp_route_near_holdout_validation.py \
  --summarize-round \
  --round-name "${ROUND_NAME}"
