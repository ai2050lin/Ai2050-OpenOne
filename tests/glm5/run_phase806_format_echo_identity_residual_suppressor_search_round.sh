#!/usr/bin/env bash
set -euo pipefail

ROUND="${1:-smoke}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

case "$ROUND" in
  smoke)
    DEFAULT_MAX_CASES=1
    MAX_ROUTES=1
    ROUTE_SIZES="6"
    TARGET_ALPHA_GRID="0.75"
    FORMAT_BETA_GRID="0,1"
    IDENTITY_BETA_GRID="0,1"
    MAX_SEMANTIC=32
    MAX_CLASSIFY=6000
    MAX_DIRECTION_SCAN=6000
    ;;
  main)
    DEFAULT_MAX_CASES=3
    MAX_ROUTES=2
    ROUTE_SIZES="6"
    TARGET_ALPHA_GRID="0.75"
    FORMAT_BETA_GRID="0,1"
    IDENTITY_BETA_GRID="0,1"
    MAX_SEMANTIC=48
    MAX_CLASSIFY=8000
    MAX_DIRECTION_SCAN=8000
    ;;
  confirm)
    DEFAULT_MAX_CASES=4
    MAX_ROUTES=2
    ROUTE_SIZES="6"
    TARGET_ALPHA_GRID="0.75"
    FORMAT_BETA_GRID="0,1"
    IDENTITY_BETA_GRID="0,1"
    MAX_SEMANTIC=64
    MAX_CLASSIFY=8000
    MAX_DIRECTION_SCAN=8000
    ;;
  *)
    echo "unknown round: $ROUND" >&2
    exit 2
    ;;
esac

for MODEL in qwen3 glm4 deepseek7b; do
  MODEL_MAX_CASES="$DEFAULT_MAX_CASES"
  if [[ "$ROUND" == "confirm" && "$MODEL" == "qwen3" ]]; then
    MODEL_MAX_CASES=5
  elif [[ "$ROUND" == "confirm" && "$MODEL" == "glm4" ]]; then
    MODEL_MAX_CASES=3
  elif [[ "$ROUND" == "confirm" && "$MODEL" == "deepseek7b" ]]; then
    MODEL_MAX_CASES=2
  fi

  COMMON_ARGS=(
    --round-name "$ROUND"
    --max-cases "$MODEL_MAX_CASES"
    --max-routes "$MAX_ROUTES"
    --route-sizes "$ROUTE_SIZES"
    --route-compare-variants "with_candidate_list,lowercase_short_value"
    --recipient-variant "without_candidate_list"
    --route-component-kinds "attn,mlp"
    --max-route-components 4
    --target-alpha-grid "$TARGET_ALPHA_GRID"
    --semantic-beta 1
    --format-beta-grid "$FORMAT_BETA_GRID"
    --identity-beta-grid "$IDENTITY_BETA_GRID"
    --semantic-direction-mode "semantic_minus_target"
    --max-semantic-new-blockers "$MAX_SEMANTIC"
    --max-residual-direction-scan "$MAX_DIRECTION_SCAN"
    --max-class-direction-tokens 64
    --max-identity-direction-tokens 64
    --max-direction-examples-saved 4
    --residual-rank-window-saved 24
    --dominant-share-threshold 0.50
    --max-near-closure-blockers 5
    --max-semantic-still-rate 0.20
    --top-k 20
    --full-rank-window 128
    --max-full-above-classify "$MAX_CLASSIFY"
    --max-baseline-blocker-classify "$MAX_CLASSIFY"
    --max-new-blocker-classify "$MAX_CLASSIFY"
    --max-surface-variants-saved 32
    --attn-implementations "flash_attention_2,sdpa,eager"
    --log-every 1
  )

  echo "[$(date +%H:%M:%S)] phase806 $ROUND: start $MODEL cases=$MODEL_MAX_CASES"
  python tests/glm5/phase806_format_echo_identity_residual_suppressor_search.py \
    --model "$MODEL" \
    "${COMMON_ARGS[@]}"
  echo "[$(date +%H:%M:%S)] phase806 $ROUND: done $MODEL"
done

python tests/glm5/phase806_format_echo_identity_residual_suppressor_search.py \
  --round-name "$ROUND" \
  --summarize-only
