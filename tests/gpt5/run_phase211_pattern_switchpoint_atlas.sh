#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-pattern_switchpoint_atlas}"
PHASE210_ROUND="${2:-minimal_pattern_transition_atlas}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

python tests/gpt5/phase211_pattern_switchpoint_atlas.py \
  --round-name "${ROUND_NAME}" \
  --phase210-round "${PHASE210_ROUND}"
