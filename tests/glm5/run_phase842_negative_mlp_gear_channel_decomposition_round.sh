#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-smoke}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

case "${ROUND_NAME}" in
  smoke)
    MAX_CASES=1
    CASE_SCOPE="candidate"
    MAX_CHANNELS=4
    SEARCH_DONORS="natural_question"
    MAX_NEW_TOKENS=6
    MAX_SPAN_CANDIDATES=32
    ;;
  main)
    MAX_CASES=1
    CASE_SCOPE="candidate"
    MAX_CHANNELS=16
    SEARCH_DONORS="natural_question,object_only"
    MAX_NEW_TOKENS=8
    MAX_SPAN_CANDIDATES=48
    ;;
  confirm)
    MAX_CASES=1
    CASE_SCOPE="candidate"
    MAX_CHANNELS=16
    SEARCH_DONORS="natural_question,object_only,natural_category,exact_choices"
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
  --max-candidates 2
  --max-negative-components 1
  --max-cases "${MAX_CASES}"
  --max-channels "${MAX_CHANNELS}"
  --search-donor-prompts "${SEARCH_DONORS}"
  --max-source-rows 0
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --attn-implementations flash_attention_2,sdpa,eager
  --log-every 1
  --max-span-candidates "${MAX_SPAN_CANDIDATES}"
  --batch-size 16
  --top-k 5
)

for MODEL in qwen3 glm4 deepseek7b; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 842 ${ROUND_NAME}: start ${MODEL}"
  python tests/glm5/phase842_negative_mlp_gear_channel_decomposition.py \
    --model "${MODEL}" \
    "${COMMON_ARGS[@]}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 842 ${ROUND_NAME}: done ${MODEL}"
done

python tests/glm5/phase842_negative_mlp_gear_channel_decomposition.py \
  --summarize-only \
  --round-name "${ROUND_NAME}"
