#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ROUND="${1:-confirm}"

echo "[$(date +%H:%M:%S)] phase820: answer boundary standard v1 round=$ROUND"
python tests/glm5/phase820_answer_boundary_standard_v1.py \
  --round-name "$ROUND"
echo "[$(date +%H:%M:%S)] phase820: done"
