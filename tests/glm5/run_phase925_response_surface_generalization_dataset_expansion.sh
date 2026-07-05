#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

ROUND_NAME="${1:-response_surface_generalization_dataset_expansion}"
COMMON_ARGS=(
  --round-name "${ROUND_NAME}"
  --phase914-round l4_mlp_route_near_holdout_validation
  --phase915-round near_boundary_action_gate_search
  --phase924-round route_protocol_response_surface_audit
  --target-blocker-token a
  --near-margin-min -2.0
  --near-margin-max 0.5
  --max-eos-rank 50
  --max-seeds-per-model 96
  --min-seeds-per-model 36
  --max-per-case 10
  --max-per-domain 40
  --max-per-group 36
)

for MODEL in qwen3 glm4 deepseek7b; do
  python tests/glm5/phase925_response_surface_generalization_dataset_expansion.py \
    --model "${MODEL}" \
    "${COMMON_ARGS[@]}"
done

python tests/glm5/phase925_response_surface_generalization_dataset_expansion.py \
  --summarize-round \
  --round-name "${ROUND_NAME}"
