#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-smoke}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

case "${ROUND_NAME}" in
  smoke)
    GEOMETRY_OBJECTS="triangle"
    MAX_CASES=1
    PROMPTS="natural_question"
    TOP_GEARS=4
    MIN_CUT_GEARS=1
    FOCUS_COMBOS=1
    COMBO_TYPES="single,pair"
    MAX_PAIRS=6
    MAX_TRIPLETS=0
    MAX_NEW_TOKENS=5
    ;;
  main)
    GEOMETRY_OBJECTS="triangle,square,rectangle,circle,polygon"
    MAX_CASES=5
    PROMPTS="natural_question,object_only,natural_category"
    TOP_GEARS=8
    MIN_CUT_GEARS=2
    FOCUS_COMBOS=4
    COMBO_TYPES="single,pair,triplet"
    MAX_PAIRS=18
    MAX_TRIPLETS=8
    MAX_NEW_TOKENS=6
    ;;
  confirm)
    GEOMETRY_OBJECTS="triangle,square,rectangle,circle,polygon"
    MAX_CASES=5
    PROMPTS="natural_question,object_only,natural_category"
    TOP_GEARS=10
    MIN_CUT_GEARS=3
    FOCUS_COMBOS=8
    COMBO_TYPES="single,pair,triplet"
    MAX_PAIRS=24
    MAX_TRIPLETS=12
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
  --phase851-round confirm
  --include-seed-triangle
  --geometry-objects "${GEOMETRY_OBJECTS}"
  --max-cases "${MAX_CASES}"
  --prompt-variants "${PROMPTS}"
  --top-gears "${TOP_GEARS}"
  --max-min-cut-gears "${MIN_CUT_GEARS}"
  --max-focus-combos "${FOCUS_COMBOS}"
  --combo-types "${COMBO_TYPES}"
  --max-pairs "${MAX_PAIRS}"
  --max-triplets "${MAX_TRIPLETS}"
  --edit-modes zero,flip
  --split-types in_sample,object_holdout,prompt_holdout
  --interaction-threshold 0.5
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --topk-entropy 20
  --attn-implementations flash_attention_2,sdpa,eager
  --log-every 1
)

echo "[Phase853] round=${ROUND_NAME} objects=${GEOMETRY_OBJECTS} prompts=${PROMPTS} top_gears=${TOP_GEARS} pairs=${MAX_PAIRS} triplets=${MAX_TRIPLETS}"

for MODEL in qwen3 glm4 deepseek7b; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 853 ${ROUND_NAME}: start ${MODEL}"
  python tests/glm5/phase853_strong_edge_expansion_natural_closure_validation.py \
    --model "${MODEL}" \
    "${COMMON_ARGS[@]}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 853 ${ROUND_NAME}: done ${MODEL}"
done

python tests/glm5/phase853_strong_edge_expansion_natural_closure_validation.py \
  --summarize-only \
  --round-name "${ROUND_NAME}"
