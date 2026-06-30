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
    ALPHA_GRID="0,0.5,1.0"
    ;;
  main)
    MAX_CASES=4
    MAX_ROUTES=2
    ROUTE_SIZES="6"
    ALPHA_GRID="0,0.25,0.5,0.75,1.0"
    ;;
  confirm)
    MAX_CASES=6
    MAX_ROUTES=3
    ROUTE_SIZES="6,8"
    ALPHA_GRID="0,0.25,0.5,0.75,1.0,1.25"
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
  --alpha-grid "$ALPHA_GRID"
  --target-gain-budget 1.0
  --min-old-suppression 0.25
  --max-stable-new-rate 0.10
  --min-anchor-improvement 0.0
  --target-boost-threshold 1.0
  --top-k 20
  --full-rank-window 128
  --max-full-above-classify 40000
  --max-baseline-blocker-classify 40000
  --max-new-blocker-classify 40000
  --max-surface-variants-saved 32
  --attn-implementations "flash_attention_2,sdpa,eager"
  --log-every 1
)

for MODEL in qwen3 glm4 deepseek7b; do
  echo "[$(date +%H:%M:%S)] phase802 $ROUND: start $MODEL"
  python tests/glm5/phase802_new_blocker_stabilization_dose_response.py \
    --model "$MODEL" \
    "${COMMON_ARGS[@]}"
  echo "[$(date +%H:%M:%S)] phase802 $ROUND: done $MODEL"
done

python tests/glm5/phase802_new_blocker_stabilization_dose_response.py \
  --round-name "$ROUND" \
  --summarize-only
