#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-smoke}"

case "${ROUND_NAME}" in
  smoke)
    MAX_SOURCE_ROWS=2
    BUDGETS="16"
    MAX_NEW_TOKENS=6
    ;;
  main)
    MAX_SOURCE_ROWS=8
    BUDGETS="16,64"
    MAX_NEW_TOKENS=8
    ;;
  confirm)
    MAX_SOURCE_ROWS=0
    BUDGETS="16,64,256"
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
  --max-source-rows "${MAX_SOURCE_ROWS}"
  --budgets "${BUDGETS}"
  --subspace-modes all,positive_topk,negative_topk,abs_topk,random_topk
  --recipient-prompt no_choices
  --donor-prompt exact_choices
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --attn-implementations flash_attention_2,sdpa,eager
  --log-every 1
)

for MODEL in qwen3 glm4 deepseek7b; do
  python tests/glm5/phase823_beneficial_harmful_boundary_subspace_split.py \
    --model "${MODEL}" \
    "${COMMON_ARGS[@]}"
done

python tests/glm5/phase823_beneficial_harmful_boundary_subspace_split.py \
  --summarize-only \
  --round-name "${ROUND_NAME}"
