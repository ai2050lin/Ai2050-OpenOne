#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

ROUND_NAME="${ROUND_NAME:-closure_quality_expansion_scan}"
MAX_CASES_PER_MODEL="${MAX_CASES_PER_MODEL:-9}"
ROLLOUT_TOKENS="${ROLLOUT_TOKENS:-32}"
MIN_BEHAVIOR="${MIN_BEHAVIOR:-0.5}"

python3 tests/gpt5/phase285_closure_quality_expansion_scan.py --model qwen3 --round-name "$ROUND_NAME" --max-cases-per-model "$MAX_CASES_PER_MODEL" --rollout-tokens "$ROLLOUT_TOKENS" --min-behavior "$MIN_BEHAVIOR"
python3 tests/gpt5/phase285_closure_quality_expansion_scan.py --model glm4 --round-name "$ROUND_NAME" --max-cases-per-model "$MAX_CASES_PER_MODEL" --rollout-tokens "$ROLLOUT_TOKENS" --min-behavior "$MIN_BEHAVIOR"
python3 tests/gpt5/phase285_closure_quality_expansion_scan.py --model deepseek7b --round-name "$ROUND_NAME" --max-cases-per-model "$MAX_CASES_PER_MODEL" --rollout-tokens "$ROLLOUT_TOKENS" --min-behavior "$MIN_BEHAVIOR"
python3 tests/gpt5/phase285_closure_quality_expansion_scan.py --round-name "$ROUND_NAME" --summarize
