#!/usr/bin/env bash
set -euo pipefail

ROUND="${1:-smoke}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

case "$ROUND" in
  smoke)
    MAX_CASES=1
    MAX_ROUTES=1
    ROUTE_SIZES="6"
    TARGET_ALPHA_GRID="0,0.75"
    SEMANTIC_BETA_GRID="0,1"
    MAX_SEMANTIC=32
    ;;
  main)
    MAX_CASES=3
    MAX_ROUTES=2
    ROUTE_SIZES="6"
    TARGET_ALPHA_GRID="0,0.75"
    SEMANTIC_BETA_GRID="0,0.5,1,1.5"
    MAX_SEMANTIC=48
    ;;
  confirm)
    MAX_CASES=5
    MAX_ROUTES=2
    ROUTE_SIZES="6"
    TARGET_ALPHA_GRID="0,0.75"
    SEMANTIC_BETA_GRID="0,1"
    MAX_SEMANTIC=64
    ;;
  *)
    echo "unknown round: $ROUND" >&2
    exit 2
    ;;
esac

COMMON_ARGS=(
  --round-name "$ROUND"
  --max-cases "$MAX_CASES"
  --max-routes "$MAX_ROUTES"
  --route-sizes "$ROUTE_SIZES"
  --route-compare-variants "with_candidate_list,lowercase_short_value"
  --recipient-variant "without_candidate_list"
  --route-component-kinds "attn,mlp"
  --max-route-components 4
  --target-alpha-grid "$TARGET_ALPHA_GRID"
  --semantic-beta-grid "$SEMANTIC_BETA_GRID"
  --semantic-direction-mode "semantic_minus_target"
  --max-semantic-new-blockers "$MAX_SEMANTIC"
  --min-true-semantic-suppression 0.20
  --max-semantic-still-rate 0.20
  --max-target-gain-delta 0.50
  --min-old-suppression 0.25
  --top-k 20
  --full-rank-window 128
  --max-full-above-classify 12000
  --max-baseline-blocker-classify 12000
  --max-new-blocker-classify 12000
  --max-surface-variants-saved 32
  --attn-implementations "flash_attention_2,sdpa,eager"
  --log-every 1
)

for MODEL in qwen3 glm4 deepseek7b; do
  echo "[$(date +%H:%M:%S)] phase804 $ROUND: start $MODEL"
  python tests/glm5/phase804_true_semantic_suppressor_projection_search.py \
    --model "$MODEL" \
    "${COMMON_ARGS[@]}"
  echo "[$(date +%H:%M:%S)] phase804 $ROUND: done $MODEL"
done

python tests/glm5/phase804_true_semantic_suppressor_projection_search.py \
  --round-name "$ROUND" \
  --summarize-only
