#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-smoke}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

case "${ROUND_NAME}" in
  smoke)
    MAX_SOURCE_ROWS=2
    BUDGETS="16"
    BASE_POOL=12
    EXPANDED_POOL=24
    EXTRA_TOP_DELTA=8
    EXTRA_TOP_SIGNED=8
    EXTRA_RANDOM=8
    GROUP_SIZE=4
    GREEDY_STEPS=3
    MAX_NEW_TOKENS=6
    VALIDATION_DONORS="natural_category"
    SAVE_INDICES_LIMIT=128
    ;;
  main)
    MAX_SOURCE_ROWS=8
    BUDGETS="16,64"
    BASE_POOL=24
    EXPANDED_POOL=48
    EXTRA_TOP_DELTA=16
    EXTRA_TOP_SIGNED=16
    EXTRA_RANDOM=16
    GROUP_SIZE=4
    GREEDY_STEPS=6
    MAX_NEW_TOKENS=8
    VALIDATION_DONORS="natural_category,natural_question"
    SAVE_INDICES_LIMIT=256
    ;;
  confirm)
    MAX_SOURCE_ROWS=0
    BUDGETS="16,64,256"
    BASE_POOL=32
    EXPANDED_POOL=72
    EXTRA_TOP_DELTA=24
    EXTRA_TOP_SIGNED=24
    EXTRA_RANDOM=24
    GROUP_SIZE=4
    GREEDY_STEPS=8
    MAX_NEW_TOKENS=8
    VALIDATION_DONORS="natural_category,natural_question,object_only"
    SAVE_INDICES_LIMIT=512
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
  --search-donor-prompt exact_choices
  --validation-donor-prompts "${VALIDATION_DONORS}"
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
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 825 ${ROUND_NAME}: start ${MODEL}"
  python tests/glm5/phase825_candidate_pool_expansion_natural_route_validation.py \
    --model "${MODEL}" \
    "${COMMON_ARGS[@]}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 825 ${ROUND_NAME}: done ${MODEL}"
done

python tests/glm5/phase825_candidate_pool_expansion_natural_route_validation.py \
  --summarize-only \
  --round-name "${ROUND_NAME}"
