#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ROUND_NAME="${1:-weak_no_single_closure_rollout}"

python tests/glm5/phase894_weak_no_single_closure_rollout_probe.py --model qwen3 --round-name "$ROUND_NAME"
python tests/glm5/phase894_weak_no_single_closure_rollout_probe.py --model glm4 --round-name "$ROUND_NAME"
python tests/glm5/phase894_weak_no_single_closure_rollout_probe.py --model deepseek7b --round-name "$ROUND_NAME"
python tests/glm5/phase894_weak_no_single_closure_rollout_probe.py --round-name "$ROUND_NAME" --summarize-round
