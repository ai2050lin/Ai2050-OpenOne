#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-smoke}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

case "${ROUND_NAME}" in
  smoke)
    CASE_SCOPE="candidate"
    MAX_CASES=1
    PROMPT_VARIANTS="natural_question"
    EDIT_MODES="original,zero,flip"
    MAX_NEW_TOKENS=6
    ;;
  main)
    CASE_SCOPE="candidate"
    MAX_CASES=1
    PROMPT_VARIANTS="natural_question,object_only,natural_category,exact_choices"
    EDIT_MODES="original,zero,flip,half"
    MAX_NEW_TOKENS=8
    ;;
  confirm)
    CASE_SCOPE="candidate_plus_geometry"
    MAX_CASES=5
    PROMPT_VARIANTS="natural_question,object_only,natural_category"
    EDIT_MODES="original,zero,flip,half"
    MAX_NEW_TOKENS=8
    ;;
  *)
    echo "unknown round: ${ROUND_NAME}" >&2
    exit 2
    ;;
esac

COMMON_ARGS=(
  --round-name "${ROUND_NAME}"
  --max-core-channels 1
  --case-scope "${CASE_SCOPE}"
  --geometry-objects triangle,square,rectangle,circle,polygon
  --max-cases "${MAX_CASES}"
  --prompt-variants "${PROMPT_VARIANTS}"
  --edit-modes "${EDIT_MODES}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --attn-implementations flash_attention_2,sdpa,eager
  --log-every 1
)

for MODEL in qwen3 glm4 deepseek7b; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 843 ${ROUND_NAME}: start ${MODEL}"
  python tests/glm5/phase843_core_channel_natural_route_validation.py \
    --model "${MODEL}" \
    "${COMMON_ARGS[@]}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 843 ${ROUND_NAME}: done ${MODEL}"
done

python tests/glm5/phase843_core_channel_natural_route_validation.py \
  --summarize-only \
  --round-name "${ROUND_NAME}"
