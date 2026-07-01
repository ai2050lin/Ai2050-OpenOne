#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-smoke}"

case "${ROUND_NAME}" in
  smoke)
    MAX_CASES=2
    LAYERS="last2"
    MAX_LAYERS=2
    MAX_NEW_TOKENS=6
    ;;
  main)
    MAX_CASES=5
    LAYERS="last4"
    MAX_LAYERS=4
    MAX_NEW_TOKENS=8
    ;;
  confirm)
    MAX_CASES=8
    LAYERS="spread"
    MAX_LAYERS=6
    MAX_NEW_TOKENS=8
    ;;
  *)
    echo "unknown round: ${ROUND_NAME}" >&2
    exit 2
    ;;
esac

COMMON_ARGS=(
  --round-name "${ROUND_NAME}"
  --source-round confirm
  --recipient-prompt no_choices
  --donor-prompt exact_choices
  --only-unclosed
  --max-cases "${MAX_CASES}"
  --layers "${LAYERS}"
  --max-layers "${MAX_LAYERS}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --patch-alpha 1.0
  --attn-implementations flash_attention_2,sdpa,eager
  --log-every 1
)

for MODEL in qwen3 glm4 deepseek7b; do
  python tests/glm5/phase821_boundary_standard_guided_causal_localization.py \
    --model "${MODEL}" \
    "${COMMON_ARGS[@]}"
done

python tests/glm5/phase821_boundary_standard_guided_causal_localization.py \
  --summarize-only \
  --round-name "${ROUND_NAME}"
