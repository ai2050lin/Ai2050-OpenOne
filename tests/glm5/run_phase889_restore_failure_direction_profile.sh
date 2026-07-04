#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ROUND_NAME="${1:-direction_set_probe}"

python tests/glm5/phase889_restore_failure_direction_profile.py --model qwen3 --round-name "$ROUND_NAME"
python tests/glm5/phase889_restore_failure_direction_profile.py --model glm4 --round-name "$ROUND_NAME"
python tests/glm5/phase889_restore_failure_direction_profile.py --model deepseek7b --round-name "$ROUND_NAME"
python tests/glm5/phase889_restore_failure_direction_profile.py --round-name "$ROUND_NAME" --summarize-round
