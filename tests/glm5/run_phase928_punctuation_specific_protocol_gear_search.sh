#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

ROUND_NAME="${1:-punctuation_specific_protocol_gear_search}"
COMMON_ARGS=(
  --round-name "${ROUND_NAME}"
  --phase926-round generalized_route_protocol_surface_validation
  --max-punctuation-seeds 12
  --coordinate-pairs 1.0:1.0,0.875:1.1,1.25:1.1,0.875:0.85,1.25:0.85,1.375:0.85,1.375:0.9
  --protocol-span-kind last8_before_period
  --target-layer 39
  --max-prefix-tokens 5
  --scale-up-factor 2.0
  --l4-candidate-pool 512
  --channel-candidate-pool 768
  --band-size 32
  --up-groups eos_support_64,margin_support_pos_64
  --up-factors 1.25,1.5,2.0
  --down-groups a_blocker_support_64,a_logit_support_64,margin_support_neg_64,band_blocker_support_64
  --down-factors 0.0,0.25,0.5,0.75
  --general-groups top_abs_64,low_abs_64
  --general-factors 0.5,1.5
  --log-every 2
)

for MODEL in qwen3 glm4 deepseek7b; do
  python tests/glm5/phase928_punctuation_specific_protocol_gear_search.py \
    --model "${MODEL}" \
    "${COMMON_ARGS[@]}"
done

python tests/glm5/phase928_punctuation_specific_protocol_gear_search.py \
  --summarize-round \
  --round-name "${ROUND_NAME}"
