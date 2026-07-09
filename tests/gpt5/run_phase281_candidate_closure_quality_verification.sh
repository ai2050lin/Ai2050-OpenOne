#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

ROUND_NAME="${ROUND_NAME:-candidate_closure_quality_verification}"
ROLLOUT_TOKENS="${ROLLOUT_TOKENS:-16}"

python3 tests/gpt5/phase281_candidate_closure_quality_verification.py --model qwen3 --round-name "$ROUND_NAME" --rollout-tokens "$ROLLOUT_TOKENS"
python3 tests/gpt5/phase281_candidate_closure_quality_verification.py --model glm4 --round-name "$ROUND_NAME" --rollout-tokens "$ROLLOUT_TOKENS"
python3 tests/gpt5/phase281_candidate_closure_quality_verification.py --model deepseek7b --round-name "$ROUND_NAME" --rollout-tokens "$ROLLOUT_TOKENS"
python3 tests/gpt5/phase281_candidate_closure_quality_verification.py --round-name "$ROUND_NAME" --summarize
