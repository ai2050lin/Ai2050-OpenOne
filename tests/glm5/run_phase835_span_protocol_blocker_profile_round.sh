#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-smoke}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

case "${ROUND_NAME}" in
  smoke)
    MAX_GROUPS=8
    MAX_PAIRS=3
    BUDGETS="16"
    SEARCH_DONORS="exact_choices,natural_category"
    MIN_NATURAL_TARGETS=1
    MAX_NEW_TOKENS=6
    MAX_SPAN_CANDIDATES=32
    PREDICTORS="category_nonresidual_else_count_nonnegative,category_count_rank_improved,category_count_span_margin_improved,category_count_span_or_rank_improved,oracle_route_target_only"
    ;;
  main)
    MAX_GROUPS=16
    MAX_PAIRS=8
    BUDGETS="16,32"
    SEARCH_DONORS="exact_choices,natural_category,object_only"
    MIN_NATURAL_TARGETS=2
    MAX_NEW_TOKENS=8
    MAX_SPAN_CANDIDATES=48
    PREDICTORS="category_nonresidual_else_count_nonnegative,category_count_rank_improved,category_count_span_margin_improved,category_count_span_rank_improved,category_count_span_closure,category_count_span_or_rank_improved,category_count_span_contrast_cleared,category_count_span_generic_cleared,oracle_route_target_only"
    ;;
  confirm)
    MAX_GROUPS=20
    MAX_PAIRS=12
    BUDGETS="16,32"
    SEARCH_DONORS="exact_choices,natural_category,natural_question,object_only"
    MIN_NATURAL_TARGETS=2
    MAX_NEW_TOKENS=8
    MAX_SPAN_CANDIDATES=48
    PREDICTORS="category_nonresidual_else_count_nonnegative,category_count_rank_improved,category_count_span_margin_improved,category_count_span_rank_improved,category_count_span_closure,category_count_span_or_rank_improved,category_count_span_and_rank_improved,category_count_span_contrast_cleared,category_count_span_generic_cleared,oracle_route_target_only"
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
  --min-natural-targets "${MIN_NATURAL_TARGETS}"
  --budgets "${BUDGETS}"
  --max-component-groups "${MAX_GROUPS}"
  --max-pairs "${MAX_PAIRS}"
  --include-weak-groups
  --max-source-rows 0
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --attn-implementations flash_attention_2,sdpa,eager
  --log-every 1
  --require-single-safe
  --max-single-degraded 0
  --require-pair-anchor
  --predictor-modes "${PREDICTORS}"
  --max-span-candidates "${MAX_SPAN_CANDIDATES}"
  --batch-size 16
  --top-k 5
)

for MODEL in qwen3 glm4 deepseek7b; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 835 ${ROUND_NAME}: start ${MODEL}"
  python tests/glm5/phase835_span_protocol_blocker_profile.py \
    --model "${MODEL}" \
    "${COMMON_ARGS[@]}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 835 ${ROUND_NAME}: done ${MODEL}"
done

python tests/glm5/phase835_span_protocol_blocker_profile.py \
  --summarize-only \
  --round-name "${ROUND_NAME}"
