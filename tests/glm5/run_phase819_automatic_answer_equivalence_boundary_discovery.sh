#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ROUNDS="${1:-smoke,main,confirm}"
PRIMARY_ROUND="${2:-confirm}"

echo "[$(date +%H:%M:%S)] phase819: boundary discovery rounds=$ROUNDS primary=$PRIMARY_ROUND"
python tests/glm5/phase819_automatic_answer_equivalence_boundary_discovery.py \
  --rounds "$ROUNDS" \
  --primary-round "$PRIMARY_ROUND"
echo "[$(date +%H:%M:%S)] phase819: done"
