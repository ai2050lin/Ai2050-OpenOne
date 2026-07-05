#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

ROUND_NAME="${1:-blocker_class_split_response_surface_audit}"
COMMON_ARGS=(
  --round-name "${ROUND_NAME}"
  --phase926-round generalized_route_protocol_surface_validation
)

for MODEL in qwen3 glm4 deepseek7b; do
  python tests/glm5/phase927_blocker_class_split_response_surface_audit.py \
    --model "${MODEL}" \
    "${COMMON_ARGS[@]}"
done

python tests/glm5/phase927_blocker_class_split_response_surface_audit.py \
  --summarize-round \
  --round-name "${ROUND_NAME}"
