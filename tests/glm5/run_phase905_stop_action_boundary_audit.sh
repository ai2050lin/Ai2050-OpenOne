#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ROUND_NAME="${1:-stop_action_boundary_audit}"

python tests/glm5/phase905_stop_action_boundary_audit.py \
  --phase904-round termination_control_candidate_search \
  --round-name "${ROUND_NAME}"
