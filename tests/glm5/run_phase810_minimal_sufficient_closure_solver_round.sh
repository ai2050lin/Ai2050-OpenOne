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
    MAX_HEADS=2
    MAX_MLP_CHANNELS=3
    MAX_COMBO_SIZE=2
    MAX_COMBO_CANDIDATES=6
    MAX_COMBOS=24
    COMBO_SCALE_GRID="1.0"
    MAX_SEMANTIC=20
    MAX_CLASSIFY=3000
    MAX_DIRECTION_SCAN=3000
    MAX_TRANSITION_CLASSIFY=3000
    ;;
  main)
    DEFAULT_MAX_CASES=2
    MAX_ROUTES=2
    ROUTE_SIZES="6"
    MAX_HEADS=3
    MAX_MLP_CHANNELS=5
    MAX_COMBO_SIZE=2
    MAX_COMBO_CANDIDATES=8
    MAX_COMBOS=48
    COMBO_SCALE_GRID="0.75,1.0"
    MAX_SEMANTIC=40
    MAX_CLASSIFY=5000
    MAX_DIRECTION_SCAN=5000
    MAX_TRANSITION_CLASSIFY=5000
    ;;
  confirm)
    DEFAULT_MAX_CASES=2
    MAX_ROUTES=2
    ROUTE_SIZES="6"
    MAX_HEADS=4
    MAX_MLP_CHANNELS=6
    MAX_COMBO_SIZE=3
    MAX_COMBO_CANDIDATES=10
    MAX_COMBOS=80
    COMBO_SCALE_GRID="0.5,0.75,1.0"
    MAX_SEMANTIC=56
    MAX_CLASSIFY=6000
    MAX_DIRECTION_SCAN=6000
    MAX_TRANSITION_CLASSIFY=6000
    ;;
  *)
    echo "unknown round: $ROUND" >&2
    exit 2
    ;;
esac

for MODEL in qwen3 glm4 deepseek7b; do
  MODEL_MAX_CASES="$DEFAULT_MAX_CASES"
  if [[ "$ROUND" == "confirm" && "$MODEL" == "qwen3" ]]; then
    MODEL_MAX_CASES=3
  elif [[ "$ROUND" == "confirm" && "$MODEL" == "glm4" ]]; then
    MODEL_MAX_CASES=2
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
    --target-alpha-grid "0.75"
    --semantic-beta 1
    --format-beta 1
    --identity-beta 0
    --include-identity-anchor
    --identity-anchor-beta-grid "0.5,1.0"
    --semantic-direction-mode "semantic_minus_target"
    --max-heads-per-component "$MAX_HEADS"
    --max-mlp-channels-per-component "$MAX_MLP_CHANNELS"
    --max-combo-size "$MAX_COMBO_SIZE"
    --max-combo-candidates "$MAX_COMBO_CANDIDATES"
    --max-combos-per-case-route "$MAX_COMBOS"
    --combo-scale-grid "$COMBO_SCALE_GRID"
    --objective-lambda-l0 0.25
    --objective-mu-new 0.2
    --objective-eta-margin 0.5
    --max-semantic-new-blockers "$MAX_SEMANTIC"
    --max-residual-direction-scan "$MAX_DIRECTION_SCAN"
    --max-transition-classify "$MAX_TRANSITION_CLASSIFY"
    --max-class-direction-tokens 64
    --max-identity-direction-tokens 64
    --max-direction-examples-saved 4
    --max-emergence-rate-for-unit 0.15
    --min-loo-net-loss 3
    --min-loo-bias-loss 0.05
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

  echo "[$(date +%H:%M:%S)] phase810 $ROUND: start $MODEL cases=$MODEL_MAX_CASES combo_size=$MAX_COMBO_SIZE pool=$MAX_COMBO_CANDIDATES combos=$MAX_COMBOS"
  python tests/glm5/phase810_minimal_sufficient_closure_solver.py \
    --model "$MODEL" \
    "${COMMON_ARGS[@]}"
  echo "[$(date +%H:%M:%S)] phase810 $ROUND: done $MODEL"
done

python tests/glm5/phase810_minimal_sufficient_closure_solver.py \
  --round-name "$ROUND" \
  --summarize-only
