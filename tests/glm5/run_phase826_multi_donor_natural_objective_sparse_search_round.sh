#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-smoke}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

case "${ROUND_NAME}" in
  smoke)
    MAX_SOURCE_ROWS=2
    BUDGETS="16"
    BASE_POOL=10
    EXPANDED_POOL=20
    EXTRA_TOP_DELTA=8
    EXTRA_TOP_SIGNED=8
    EXTRA_RANDOM=8
    GROUP_SIZE=4
    GREEDY_STEPS=2
    MAX_NEW_TOKENS=6
    SEARCH_DONORS="exact_choices,natural_category"
    SAVE_INDICES_LIMIT=128
    ;;
  main)
    MAX_SOURCE_ROWS=4
    BUDGETS="16,64"
    BASE_POOL=14
    EXPANDED_POOL=32
    EXTRA_TOP_DELTA=12
    EXTRA_TOP_SIGNED=12
    EXTRA_RANDOM=12
    GROUP_SIZE=4
    GREEDY_STEPS=4
    MAX_NEW_TOKENS=8
    SEARCH_DONORS="exact_choices,natural_category,object_only"
    SAVE_INDICES_LIMIT=256
    ;;
  confirm)
    MAX_SOURCE_ROWS=8
    BUDGETS="16,64"
    BASE_POOL=18
    EXPANDED_POOL=40
    EXTRA_TOP_DELTA=16
    EXTRA_TOP_SIGNED=16
    EXTRA_RANDOM=16
    GROUP_SIZE=4
    GREEDY_STEPS=5
    MAX_NEW_TOKENS=8
    SEARCH_DONORS="exact_choices,natural_category,natural_question,object_only"
    SAVE_INDICES_LIMIT=256
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
  --search-donor-prompts "${SEARCH_DONORS}"
  --max-source-rows "${MAX_SOURCE_ROWS}"
  --budgets "${BUDGETS}"
  --base-candidate-pool "${BASE_POOL}"
  --expanded-candidate-pool "${EXPANDED_POOL}"
  --extra-top-delta "${EXTRA_TOP_DELTA}"
  --extra-top-signed "${EXTRA_TOP_SIGNED}"
  --extra-random "${EXTRA_RANDOM}"
  --search-group-size "${GROUP_SIZE}"
  --greedy-steps "${GREEDY_STEPS}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --attn-implementations flash_attention_2,sdpa,eager
  --save-indices-limit "${SAVE_INDICES_LIMIT}"
  --log-every 1
)

for MODEL in qwen3 glm4 deepseek7b; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 826 ${ROUND_NAME}: start ${MODEL}"
  python tests/glm5/phase826_multi_donor_natural_objective_sparse_search.py \
    --model "${MODEL}" \
    "${COMMON_ARGS[@]}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 826 ${ROUND_NAME}: done ${MODEL}"
done

python tests/glm5/phase826_multi_donor_natural_objective_sparse_search.py \
  --summarize-only \
  --round-name "${ROUND_NAME}"
