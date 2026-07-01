#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-smoke}"

case "${ROUND_NAME}" in
  smoke)
    MAX_CASES=1
    MAX_HEADS=6
    MLP_GROUPS="1,8"
    MAX_NEW_TOKENS=6
    ;;
  main)
    MAX_CASES=3
    MAX_HEADS=12
    MLP_GROUPS="1,8,32"
    MAX_NEW_TOKENS=8
    ;;
  confirm)
    MAX_CASES=4
    MAX_HEADS=0
    MLP_GROUPS="1,8,32"
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
  --max-cases "${MAX_CASES}"
  --max-heads "${MAX_HEADS}"
  --mlp-channel-groups "${MLP_GROUPS}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --recipient-prompt no_choices
  --donor-prompt exact_choices
  --attn-implementations flash_attention_2,sdpa,eager
  --log-every 1
)

for MODEL in qwen3 glm4 deepseek7b; do
  python tests/glm5/phase822_boundary_transition_head_mlp_decomposition.py \
    --model "${MODEL}" \
    "${COMMON_ARGS[@]}"
done

python tests/glm5/phase822_boundary_transition_head_mlp_decomposition.py \
  --summarize-only \
  --round-name "${ROUND_NAME}"
