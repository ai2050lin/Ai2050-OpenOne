#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ROUNDS="${1:-smoke,main,confirm}"

echo "[$(date +%H:%M:%S)] phase817: audit rounds=$ROUNDS"
python tests/glm5/phase817_alias_aware_answer_span_audit.py \
  --rounds "$ROUNDS"
echo "[$(date +%H:%M:%S)] phase817: done"
