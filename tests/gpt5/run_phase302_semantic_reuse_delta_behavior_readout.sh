#!/usr/bin/env bash
set -euo pipefail

LIMIT="${1:-0}"
ROUND_NAME="${2:-semantic_reuse_delta_behavior_readout}"

cd "$(dirname "$0")/../.."

python tests/gpt5/phase302_semantic_reuse_delta_behavior_readout_runner.py --model qwen3 --limit "${LIMIT}" --round-name "${ROUND_NAME}"
python tests/gpt5/phase302_semantic_reuse_delta_behavior_readout_runner.py --model glm4 --limit "${LIMIT}" --round-name "${ROUND_NAME}"
python tests/gpt5/phase302_semantic_reuse_delta_behavior_readout_runner.py --model deepseek7b --limit "${LIMIT}" --round-name "${ROUND_NAME}"
python tests/gpt5/phase302_semantic_reuse_delta_behavior_readout_runner.py --summarize --round-name "${ROUND_NAME}"
