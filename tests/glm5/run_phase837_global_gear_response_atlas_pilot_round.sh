#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-smoke}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

case "${ROUND_NAME}" in
  smoke)
    MAX_GROUPS=4
    MAX_CASES=4
    BUDGETS="16"
    SEARCH_DONORS="exact_choices,natural_category"
    MAX_NEW_TOKENS=6
    MAX_SPAN_CANDIDATES=32
    ;;
  main)
    MAX_GROUPS=6
    MAX_CASES=8
    BUDGETS="16,32"
    SEARCH_DONORS="exact_choices,natural_category,object_only"
    MAX_NEW_TOKENS=8
    MAX_SPAN_CANDIDATES=48
    ;;
  confirm)
    MAX_GROUPS=8
    MAX_CASES=12
    BUDGETS="16,32"
    SEARCH_DONORS="exact_choices,natural_category,natural_question,object_only"
    MAX_NEW_TOKENS=8
    MAX_SPAN_CANDIDATES=48
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
  --min-natural-targets 2
  --budgets "${BUDGETS}"
  --max-component-groups "${MAX_GROUPS}"
  --max-cases "${MAX_CASES}"
  --include-weak-groups
  --max-source-rows 0
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --attn-implementations flash_attention_2,sdpa,eager
  --log-every 1
  --max-span-candidates "${MAX_SPAN_CANDIDATES}"
  --batch-size 16
  --top-k 5
)

for MODEL in qwen3 glm4 deepseek7b; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 837 ${ROUND_NAME}: start ${MODEL}"
  python tests/glm5/phase837_global_gear_response_atlas_pilot.py \
    --model "${MODEL}" \
    "${COMMON_ARGS[@]}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 837 ${ROUND_NAME}: done ${MODEL}"
done

python tests/glm5/phase837_global_gear_response_atlas_pilot.py \
  --summarize-only \
  --round-name "${ROUND_NAME}"
