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
    PREDICTORS="selected_count_nonnegative,category_nonresidual_else_count_nonnegative,oracle_route_target_only"
    ;;
  main)
    MAX_GROUPS=16
    MAX_PAIRS=8
    BUDGETS="16,32"
    SEARCH_DONORS="exact_choices,natural_category,object_only"
    MIN_NATURAL_TARGETS=2
    MAX_NEW_TOKENS=8
    PREDICTORS="selected_sum_positive,selected_count_majority,selected_count_nonnegative,full_sum_positive,full_count_majority,donor_selected_gain_positive,non_residual_count_nonnegative,category_nonresidual_else_count_nonnegative,category_nonresidual_else_sum_positive,oracle_route_target_only"
    ;;
  confirm)
    MAX_GROUPS=20
    MAX_PAIRS=12
    BUDGETS="16,32"
    SEARCH_DONORS="exact_choices,natural_category,natural_question,object_only"
    MIN_NATURAL_TARGETS=2
    MAX_NEW_TOKENS=8
    PREDICTORS="selected_sum_positive,selected_count_majority,selected_count_nonnegative,selected_energy_ratio_55,full_sum_positive,full_count_majority,donor_selected_gain_positive,donor_full_gain_positive,non_residual_count_nonnegative,category_nonresidual_else_count_nonnegative,category_nonresidual_else_sum_positive,category_nonresidual_else_gain_positive,oracle_route_target_only"
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
)

for MODEL in qwen3 glm4 deepseek7b; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 833 ${ROUND_NAME}: start ${MODEL}"
  python tests/glm5/phase833_internal_route_boundary_predictor_search.py \
    --model "${MODEL}" \
    "${COMMON_ARGS[@]}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 833 ${ROUND_NAME}: done ${MODEL}"
done

python tests/glm5/phase833_internal_route_boundary_predictor_search.py \
  --summarize-only \
  --round-name "${ROUND_NAME}"
