#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-stable_protocol_prompt_trigger_atlas}"
MAX_CASES="${MAX_CASES:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24}"

python tests/gpt5/phase239_stable_protocol_prompt_trigger_atlas.py \
  --model qwen3 \
  --round-name "${ROUND_NAME}" \
  --max-cases "${MAX_CASES}" \
  --max-new-tokens "${MAX_NEW_TOKENS}"

python tests/gpt5/phase239_stable_protocol_prompt_trigger_atlas.py \
  --model glm4 \
  --round-name "${ROUND_NAME}" \
  --max-cases "${MAX_CASES}" \
  --max-new-tokens "${MAX_NEW_TOKENS}"

python tests/gpt5/phase239_stable_protocol_prompt_trigger_atlas.py \
  --model deepseek7b \
  --round-name "${ROUND_NAME}" \
  --max-cases "${MAX_CASES}" \
  --max-new-tokens "${MAX_NEW_TOKENS}"

python tests/gpt5/phase239_stable_protocol_prompt_trigger_atlas.py \
  --summarize \
  --round-name "${ROUND_NAME}"
