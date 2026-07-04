#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ROUND_NAME="${1:-non_color_axis_discovery}"

python tests/glm5/phase897_non_color_route_axis_discovery.py --model qwen3 --round-name "$ROUND_NAME"
python tests/glm5/phase897_non_color_route_axis_discovery.py --model glm4 --round-name "$ROUND_NAME"
python tests/glm5/phase897_non_color_route_axis_discovery.py --model deepseek7b --round-name "$ROUND_NAME"
python tests/glm5/phase897_non_color_route_axis_discovery.py --round-name "$ROUND_NAME" --summarize-round
