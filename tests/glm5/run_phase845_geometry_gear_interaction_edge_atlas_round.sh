#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-smoke}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

case "${ROUND_NAME}" in
  smoke)
    GEOMETRY_OBJECTS="triangle"
    MAX_CASES=1
    PROMPTS="natural_question"
    TOP_GEARS=4
    COMBO_TYPES="single,pair"
    INCLUDE_TRIPLETS=0
    MAX_PAIRS=6
    MAX_TRIPLETS=0
    EDIT_MODES="zero,flip"
    MAX_NEW_TOKENS=5
    ;;
  main)
    GEOMETRY_OBJECTS="triangle,square,rectangle,circle"
    MAX_CASES=4
    PROMPTS="natural_question,natural_category"
    TOP_GEARS=6
    COMBO_TYPES="single,pair,triplet"
    INCLUDE_TRIPLETS=1
    MAX_PAIRS=15
    MAX_TRIPLETS=4
    EDIT_MODES="zero,flip"
    MAX_NEW_TOKENS=6
    ;;
  confirm)
    GEOMETRY_OBJECTS="triangle,square,rectangle,circle,polygon"
    MAX_CASES=5
    PROMPTS="natural_question,object_only,natural_category"
    TOP_GEARS=6
    COMBO_TYPES="single,pair,triplet"
    INCLUDE_TRIPLETS=1
    MAX_PAIRS=15
    MAX_TRIPLETS=4
    EDIT_MODES="zero,flip"
    MAX_NEW_TOKENS=6
    ;;
  *)
    echo "unknown round: ${ROUND_NAME}" >&2
    exit 2
    ;;
esac

COMMON_ARGS=(
  --round-name "${ROUND_NAME}"
  --phase844-round confirm
  --include-seed-triangle
  --geometry-objects "${GEOMETRY_OBJECTS}"
  --max-cases "${MAX_CASES}"
  --prompt-variants "${PROMPTS}"
  --top-gears "${TOP_GEARS}"
  --combo-types "${COMBO_TYPES}"
  --max-pairs "${MAX_PAIRS}"
  --max-triplets "${MAX_TRIPLETS}"
  --edit-modes "${EDIT_MODES}"
  --interaction-threshold 0.5
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --attn-implementations flash_attention_2,sdpa,eager
  --log-every 1
)

if [[ "${INCLUDE_TRIPLETS}" == "1" ]]; then
  COMMON_ARGS+=(--include-triplets)
fi

for MODEL in qwen3 glm4 deepseek7b; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 845 ${ROUND_NAME}: start ${MODEL}"
  python tests/glm5/phase845_geometry_gear_interaction_edge_atlas.py \
    --model "${MODEL}" \
    "${COMMON_ARGS[@]}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 845 ${ROUND_NAME}: done ${MODEL}"
done

python tests/glm5/phase845_geometry_gear_interaction_edge_atlas.py \
  --summarize-only \
  --round-name "${ROUND_NAME}"
