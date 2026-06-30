#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-smoke}"
shift || true

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SCRIPT="tests/glm5/phase783_token_position_surface_route_patch.py"

for MODEL in qwen3 glm4 deepseek7b; do
  echo "[phase783] $(date '+%Y-%m-%d %H:%M:%S') start ${MODEL} round=${ROUND_NAME}"
  python "$SCRIPT" --model "$MODEL" --round-name "$ROUND_NAME" "$@" --hard-exit-after-model
  echo "[phase783] $(date '+%Y-%m-%d %H:%M:%S') done ${MODEL} round=${ROUND_NAME}"
done

python "$SCRIPT" --round-name "$ROUND_NAME" --summarize-only "$@"
echo "[phase783] $(date '+%Y-%m-%d %H:%M:%S') cross summary complete round=${ROUND_NAME}"
