#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

ROUND_NAME="${1:-same_case_candidate_residual_holdout}"
COMMON_ARGS=(
  --round-name "${ROUND_NAME}"
  --phase925-round response_surface_generalization_dataset_expansion
  --phase930-round natural_gate_strict_clean_transition_audit
  --max-holdout-seeds 90
  --max-per-case 30
  --coordinate-pairs 1.0:1.0,0.875:1.1,1.25:1.1,0.875:0.85,1.25:0.85,1.375:0.85,1.375:0.9
  --factors 2.1,2.25
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
  python tests/glm5/phase936_same_case_candidate_residual_holdout.py \
    --model "${MODEL}" \
    "${COMMON_ARGS[@]}"
done

python tests/glm5/phase936_same_case_candidate_residual_holdout.py \
  --summarize-round \
  --round-name "${ROUND_NAME}"
