#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

ROUND_NAME="${1:-generalized_route_protocol_surface_validation}"
COMMON_ARGS=(
  --round-name "${ROUND_NAME}"
  --phase925-round response_surface_generalization_dataset_expansion
  --max-seeds-per-model 30
  --min-seeds-per-model 24
  --max-per-case 4
  --max-per-domain 12
  --max-per-group 12
  --max-per-blocker-class 18
  --target-layer 39
  --max-prefix-tokens 5
  --scale-up-factor 2.0
  --l4-candidate-pool 512
  --channel-candidate-pool 768
  --band-size 32
  --group-budget 64
  --l39-factors 1.25,1.375
  --route-alphas 0.75,0.875,1.0,1.125,1.25,1.375
  --protocol-span-kind last8_before_period
  --protocol-factors 0.85,0.9,1.0,1.1
  --log-every 2
)

for MODEL in qwen3 glm4 deepseek7b; do
  python tests/glm5/phase926_generalized_route_protocol_surface_validation.py \
    --model "${MODEL}" \
    "${COMMON_ARGS[@]}"
done

python tests/glm5/phase926_generalized_route_protocol_surface_validation.py \
  --summarize-round \
  --round-name "${ROUND_NAME}"
