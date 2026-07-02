#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-smoke}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

case "${ROUND_NAME}" in
  smoke)
    MAX_CANDIDATES=1
    MAX_CASES=1
    SEARCH_DONORS="natural_question"
    CASE_SCOPE="candidate"
    MAX_NEW_TOKENS=6
    MAX_SPAN_CANDIDATES=32
    ;;
  main)
    MAX_CANDIDATES=2
    MAX_CASES=4
    SEARCH_DONORS="natural_question,object_only,natural_category"
    CASE_SCOPE="candidate_plus_holdout"
    MAX_NEW_TOKENS=8
    MAX_SPAN_CANDIDATES=48
    ;;
  confirm)
    MAX_CANDIDATES=2
    MAX_CASES=4
    SEARCH_DONORS="natural_question,object_only,natural_category,exact_choices"
    CASE_SCOPE="candidate_plus_holdout"
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
  --max-source-rows 0
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --attn-implementations flash_attention_2,sdpa,eager
  --log-every 1
  --max-span-candidates "${MAX_SPAN_CANDIDATES}"
  --batch-size 16
  --top-k 5
  --interaction-quality-threshold 0.05
  --echo-tolerance 0.05
  --harm-tolerance 0.05
  --max-minimal-echo-risk 0.25
  --max-minimal-harm-risk 0.05
  --natural-positive-threshold 0.0
)

for MODEL in qwen3 glm4 deepseek7b; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 840 ${ROUND_NAME}: start ${MODEL}"
  python tests/glm5/phase840_strict_target_interaction_natural_coactivation.py \
    --model "${MODEL}" \
    "${COMMON_ARGS[@]}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 840 ${ROUND_NAME}: done ${MODEL}"
done

python tests/glm5/phase840_strict_target_interaction_natural_coactivation.py \
  --summarize-only \
  --round-name "${ROUND_NAME}"
