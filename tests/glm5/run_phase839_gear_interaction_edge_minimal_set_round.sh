#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-smoke}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

case "${ROUND_NAME}" in
  smoke)
    MAX_COMPONENTS=2
    MAX_CASES=2
    MAX_PAIRS=1
    INCLUDE_SETS=0
    SEARCH_DONORS="exact_choices,natural_category"
    MAX_NEW_TOKENS=6
    MAX_SPAN_CANDIDATES=32
    ;;
  main)
    MAX_COMPONENTS=3
    MAX_CASES=4
    MAX_PAIRS=3
    INCLUDE_SETS=1
    SEARCH_DONORS="exact_choices,natural_category,object_only"
    MAX_NEW_TOKENS=8
    MAX_SPAN_CANDIDATES=48
    ;;
  confirm)
    MAX_COMPONENTS=4
    MAX_CASES=4
    MAX_PAIRS=6
    INCLUDE_SETS=1
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
  --case-scope holdout
  --max-components "${MAX_COMPONENTS}"
  --max-cases "${MAX_CASES}"
  --max-pairs "${MAX_PAIRS}"
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
)

if [[ "${INCLUDE_SETS}" == "1" ]]; then
  COMMON_ARGS+=(--include-sets --max-set-size "${MAX_COMPONENTS}")
fi

for MODEL in qwen3 glm4 deepseek7b; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 839 ${ROUND_NAME}: start ${MODEL}"
  python tests/glm5/phase839_gear_interaction_edge_minimal_set.py \
    --model "${MODEL}" \
    "${COMMON_ARGS[@]}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 839 ${ROUND_NAME}: done ${MODEL}"
done

python tests/glm5/phase839_gear_interaction_edge_minimal_set.py \
  --summarize-only \
  --round-name "${ROUND_NAME}"
