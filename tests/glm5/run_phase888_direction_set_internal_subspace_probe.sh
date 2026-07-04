#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ROUND_NAME="${1:-direction_set_probe}"

python tests/glm5/phase888_direction_set_internal_subspace_probe.py --model qwen3 --round-name "$ROUND_NAME"
python tests/glm5/phase888_direction_set_internal_subspace_probe.py --model glm4 --round-name "$ROUND_NAME"
python tests/glm5/phase888_direction_set_internal_subspace_probe.py --model deepseek7b --round-name "$ROUND_NAME"
python tests/glm5/phase888_direction_set_internal_subspace_probe.py --round-name "$ROUND_NAME" --summarize-round
