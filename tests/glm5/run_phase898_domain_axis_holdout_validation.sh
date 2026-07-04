#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ROUND_NAME="${1:-domain_axis_holdout_validation}"

python tests/glm5/phase898_domain_axis_holdout_validation.py --model qwen3 --round-name "$ROUND_NAME"
python tests/glm5/phase898_domain_axis_holdout_validation.py --model glm4 --round-name "$ROUND_NAME"
python tests/glm5/phase898_domain_axis_holdout_validation.py --model deepseek7b --round-name "$ROUND_NAME"
python tests/glm5/phase898_domain_axis_holdout_validation.py --round-name "$ROUND_NAME" --summarize-round
