#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

ROUND_NAME="${ROUND_NAME:-fourth_gap_batch_physical_path_fill}"
MAX_CASES_PER_MODEL="${MAX_CASES_PER_MODEL:-18}"
ROLLOUT_TOKENS="${ROLLOUT_TOKENS:-6}"

python3 tests/gpt5/phase283_fourth_gap_batch_physical_path_fill.py --model qwen3 --round-name "$ROUND_NAME" --max-cases-per-model "$MAX_CASES_PER_MODEL" --rollout-tokens "$ROLLOUT_TOKENS"
python3 tests/gpt5/phase283_fourth_gap_batch_physical_path_fill.py --model glm4 --round-name "$ROUND_NAME" --max-cases-per-model "$MAX_CASES_PER_MODEL" --rollout-tokens "$ROLLOUT_TOKENS"
python3 tests/gpt5/phase283_fourth_gap_batch_physical_path_fill.py --model deepseek7b --round-name "$ROUND_NAME" --max-cases-per-model "$MAX_CASES_PER_MODEL" --rollout-tokens "$ROLLOUT_TOKENS"
python3 tests/gpt5/phase283_fourth_gap_batch_physical_path_fill.py --round-name "$ROUND_NAME" --summarize
