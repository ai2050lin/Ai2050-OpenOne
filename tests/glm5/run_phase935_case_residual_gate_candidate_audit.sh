#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

ROUND_NAME="${1:-case_residual_gate_candidate_audit}"
COMMON_ARGS=(
  --round-name "${ROUND_NAME}"
  --phase930-round natural_gate_strict_clean_transition_audit
  --phase934-round case_residual_size_control_audit
)

for MODEL in qwen3 glm4 deepseek7b; do
  python tests/glm5/phase935_case_residual_gate_candidate_audit.py \
    --model "${MODEL}" \
    "${COMMON_ARGS[@]}"
done

python tests/glm5/phase935_case_residual_gate_candidate_audit.py \
  --summarize-round \
  --round-name "${ROUND_NAME}"
