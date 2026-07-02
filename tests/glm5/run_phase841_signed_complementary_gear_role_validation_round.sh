#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-smoke}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

case "${ROUND_NAME}" in
  smoke)
    MAX_CANDIDATES=1
    MAX_CASES=1
    CASE_SCOPE="candidate"
    SEARCH_DONORS="natural_question"
    MODES="pair_original,positive_only,negative_only,flip_negative,zero_negative"
    MAX_NEW_TOKENS=6
    MAX_SPAN_CANDIDATES=32
    ;;
  main)
    MAX_CANDIDATES=2
    MAX_CASES=1
    CASE_SCOPE="candidate"
    SEARCH_DONORS="natural_question,object_only"
    MODES="pair_original,positive_only,negative_only,flip_positive,flip_negative,zero_positive,zero_negative"
    MAX_NEW_TOKENS=8
    MAX_SPAN_CANDIDATES=48
    ;;
  confirm)
    MAX_CANDIDATES=2
    MAX_CASES=4
    CASE_SCOPE="candidate_plus_holdout"
    SEARCH_DONORS="natural_question,object_only,natural_category,exact_choices"
    MODES="pair_original,positive_only,negative_only,flip_positive,flip_negative,zero_positive,zero_negative,zero_all"
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
  --case-scope "${CASE_SCOPE}"
  --candidate-combo-kinds pair
  --max-candidates "${MAX_CANDIDATES}"
  --max-cases "${MAX_CASES}"
  --search-donor-prompts "${SEARCH_DONORS}"
  --modes "${MODES}"
  --max-source-rows 0
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --attn-implementations flash_attention_2,sdpa,eager
  --log-every 1
  --max-span-candidates "${MAX_SPAN_CANDIDATES}"
  --batch-size 16
  --top-k 5
  --natural-positive-threshold 0.0
)

for MODEL in qwen3 glm4 deepseek7b; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 841 ${ROUND_NAME}: start ${MODEL}"
  python tests/glm5/phase841_signed_complementary_gear_role_validation.py \
    --model "${MODEL}" \
    "${COMMON_ARGS[@]}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 841 ${ROUND_NAME}: done ${MODEL}"
done

python tests/glm5/phase841_signed_complementary_gear_role_validation.py \
  --summarize-only \
  --round-name "${ROUND_NAME}"
