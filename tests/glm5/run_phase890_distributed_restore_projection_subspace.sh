#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ROUND_NAME="${1:-distributed_restore_projection}"

python tests/glm5/phase890_distributed_restore_projection_subspace.py --model qwen3 --round-name "$ROUND_NAME"
python tests/glm5/phase890_distributed_restore_projection_subspace.py --model glm4 --round-name "$ROUND_NAME"
python tests/glm5/phase890_distributed_restore_projection_subspace.py --model deepseek7b --round-name "$ROUND_NAME"
python tests/glm5/phase890_distributed_restore_projection_subspace.py --round-name "$ROUND_NAME" --summarize-round
