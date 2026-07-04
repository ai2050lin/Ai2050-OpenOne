#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ROUND_NAME="${1:-channel_complementarity_coordinate_basis}"

python tests/glm5/phase892_channel_complementarity_coordinate_basis_probe.py --model qwen3 --round-name "$ROUND_NAME"
python tests/glm5/phase892_channel_complementarity_coordinate_basis_probe.py --model glm4 --round-name "$ROUND_NAME"
python tests/glm5/phase892_channel_complementarity_coordinate_basis_probe.py --model deepseek7b --round-name "$ROUND_NAME"
python tests/glm5/phase892_channel_complementarity_coordinate_basis_probe.py --round-name "$ROUND_NAME" --summarize-round
