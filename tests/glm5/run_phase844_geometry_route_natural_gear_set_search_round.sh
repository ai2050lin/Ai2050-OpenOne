#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-smoke}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

case "${ROUND_NAME}" in
  smoke)
    GEOMETRY_OBJECTS="triangle"
    MAX_CASES=1
    PROMPTS="natural_question"
    LAYERS="27"
    PER_SAMPLE_TOPK=32
    MIN_HITS=1
    MAX_GEARS=4
    SUBSET_SIZES="1,4"
    EDIT_MODES="zero,flip"
    MAX_NEW_TOKENS=5
    ;;
  main)
    GEOMETRY_OBJECTS="triangle,square,rectangle,circle"
    MAX_CASES=4
    PROMPTS="natural_question,natural_category"
    LAYERS="26,27,28"
    PER_SAMPLE_TOPK=64
    MIN_HITS=2
    MAX_GEARS=8
    SUBSET_SIZES="1,4,8"
    EDIT_MODES="zero,flip"
    MAX_NEW_TOKENS=6
    ;;
  confirm)
    GEOMETRY_OBJECTS="triangle,square,rectangle,circle,polygon"
    MAX_CASES=5
    PROMPTS="natural_question,object_only,natural_category"
    LAYERS="24,25,26,27,28,29,30"
    PER_SAMPLE_TOPK=64
    MIN_HITS=2
    MAX_GEARS=12
    SUBSET_SIZES="1,4,8,12"
    EDIT_MODES="zero,flip,half"
    MAX_NEW_TOKENS=6
    ;;
  *)
    echo "unknown round: ${ROUND_NAME}" >&2
    exit 2
    ;;
esac

COMMON_ARGS=(
  --round-name "${ROUND_NAME}"
  --include-seed-triangle
  --geometry-objects "${GEOMETRY_OBJECTS}"
  --max-cases "${MAX_CASES}"
  --prompt-variants "${PROMPTS}"
  --layers "${LAYERS}"
  --per-sample-topk "${PER_SAMPLE_TOPK}"
  --min-candidate-hits "${MIN_HITS}"
  --max-gears "${MAX_GEARS}"
  --subset-sizes "${SUBSET_SIZES}"
  --edit-modes "${EDIT_MODES}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --attn-implementations flash_attention_2,sdpa,eager
  --log-every 1
)

for MODEL in qwen3 glm4 deepseek7b; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 844 ${ROUND_NAME}: start ${MODEL}"
  python tests/glm5/phase844_geometry_route_natural_gear_set_search.py \
    --model "${MODEL}" \
    "${COMMON_ARGS[@]}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 844 ${ROUND_NAME}: done ${MODEL}"
done

python tests/glm5/phase844_geometry_route_natural_gear_set_search.py \
  --summarize-only \
  --round-name "${ROUND_NAME}"
