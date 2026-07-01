#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-smoke}"

case "${ROUND_NAME}" in
  smoke)
    MAX_SOURCE_ROWS=2
    BUDGETS="16"
    CANDIDATE_POOL=12
    GROUP_SIZE=4
    GREEDY_STEPS=3
    MAX_NEW_TOKENS=6
    SAVE_PROBES=()
    ;;
  main)
    MAX_SOURCE_ROWS=8
    BUDGETS="16,64"
    CANDIDATE_POOL=24
    GROUP_SIZE=4
    GREEDY_STEPS=6
    MAX_NEW_TOKENS=8
    SAVE_PROBES=()
    ;;
  confirm)
    MAX_SOURCE_ROWS=0
    BUDGETS="16,64,256"
    CANDIDATE_POOL=32
    GROUP_SIZE=4
    GREEDY_STEPS=8
    MAX_NEW_TOKENS=8
    SAVE_PROBES=(--save-probe-rows)
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
  --candidate-pool "${CANDIDATE_POOL}"
  --search-group-size "${GROUP_SIZE}"
  --greedy-steps "${GREEDY_STEPS}"
  --reference-modes all,positive_topk,abs_topk,random_topk
  --recipient-prompt no_choices
  --donor-prompt exact_choices
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --attn-implementations flash_attention_2,sdpa,eager
  --log-every 1
  "${SAVE_PROBES[@]}"
)

for MODEL in qwen3 glm4 deepseek7b; do
  python tests/glm5/phase824_boundary_objective_sparse_subspace_search.py \
    --model "${MODEL}" \
    "${COMMON_ARGS[@]}"
done

python tests/glm5/phase824_boundary_objective_sparse_subspace_search.py \
  --summarize-only \
  --round-name "${ROUND_NAME}"
