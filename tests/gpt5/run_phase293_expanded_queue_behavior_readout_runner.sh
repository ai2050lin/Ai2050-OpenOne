#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
LIMIT="${1:-0}"
python tests/gpt5/phase293_expanded_queue_behavior_readout_runner.py --model qwen3 --limit "$LIMIT"
python tests/gpt5/phase293_expanded_queue_behavior_readout_runner.py --model glm4 --limit "$LIMIT"
python tests/gpt5/phase293_expanded_queue_behavior_readout_runner.py --model deepseek7b --limit "$LIMIT"
