#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-pattern_family_behavior_benchmark}"
MAX_CASES="${MAX_CASES:-44}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24}"

python tests/gpt5/phase236_pattern_family_behavior_benchmark.py \
  --model qwen3 \
  --round-name "${ROUND_NAME}" \
  --max-cases "${MAX_CASES}" \
  --max-new-tokens "${MAX_NEW_TOKENS}"

python tests/gpt5/phase236_pattern_family_behavior_benchmark.py \
  --model glm4 \
  --round-name "${ROUND_NAME}" \
  --max-cases "${MAX_CASES}" \
  --max-new-tokens "${MAX_NEW_TOKENS}"

python tests/gpt5/phase236_pattern_family_behavior_benchmark.py \
  --model deepseek7b \
  --round-name "${ROUND_NAME}" \
  --max-cases "${MAX_CASES}" \
  --max-new-tokens "${MAX_NEW_TOKENS}"

python tests/gpt5/phase236_pattern_family_behavior_benchmark.py \
  --summarize \
  --round-name "${ROUND_NAME}"
