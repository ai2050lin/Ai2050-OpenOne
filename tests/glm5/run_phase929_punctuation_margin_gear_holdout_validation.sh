#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

ROUND_NAME="${1:-punctuation_margin_gear_holdout_validation}"
COMMON_ARGS=(
  --round-name "${ROUND_NAME}"
  --phase925-round response_surface_generalization_dataset_expansion
  --phase928-round punctuation_specific_protocol_gear_search
  --seed-source selected
  --max-punctuation-seeds 30
  --max-per-case 10
  --coordinate-pairs 1.0:1.0,0.875:1.1,1.25:1.1,0.875:0.85,1.25:0.85,1.375:0.85,1.375:0.9
  --margin-factors 1.25,1.5,1.75,2.0,2.25
  --eos-control-factors 2.0
  --blocker-control-factors 0.25
  --protocol-span-kind last8_before_period
  --target-layer 39
  --max-prefix-tokens 5
  --scale-up-factor 2.0
  --l4-candidate-pool 512
  --channel-candidate-pool 768
  --band-size 32
  --log-every 5
)

for MODEL in qwen3 glm4 deepseek7b; do
  python tests/glm5/phase929_punctuation_margin_gear_holdout_validation.py \
    --model "${MODEL}" \
    "${COMMON_ARGS[@]}"
done

python tests/glm5/phase929_punctuation_margin_gear_holdout_validation.py \
  --summarize-round \
  --round-name "${ROUND_NAME}"
