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
    IDENTITY_BETA_GRID="0.5,1.0,1.5"
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
    IDENTITY_BETA_GRID="0.25,0.5,1.0,1.5,2.0"
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
    IDENTITY_BETA_GRID="0.25,0.5,1.0,1.5,2.0,3.0"
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
    --identity-anchor-beta-grid "$IDENTITY_BETA_GRID"
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
    --objective-unreduced-class-penalty 2.0
    --objective-class-resolution-bonus 0.1
    --objective-answer-unit-weight 8.0
    --objective-answer-unit-margin-weight 1.0
    --objective-answer-class-preserve-weight 50.0
    --objective-answer-class-weight 5.0
    --objective-contrast-class-weight 3.0
    --objective-surface-fragment-weight 0.5
    --objective-class-weighted-after-scale 0.25
    --min-class-coverage-rate 0.35
    --answer-variant-scan-topk 768
    --max-answer-equiv-ids-saved 32
    --class-weight-overrides "candidate_list_or_case_value:1.8,designated_contrast:1.6,semantic_or_lexical_competitor:1.35,echo_token:1.25,high_frequency_or_format:1.15,whitespace_or_newline:1.05,punctuation:0.95,number_or_symbol:0.9"
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

  echo "[$(date +%H:%M:%S)] phase814 $ROUND: start $MODEL cases=$MODEL_MAX_CASES combo_size=$MAX_COMBO_SIZE pool=$MAX_COMBO_CANDIDATES combos=$MAX_COMBOS"
  python tests/glm5/phase814_tokenizer_valid_answer_unit_closure.py \
    --model "$MODEL" \
    "${COMMON_ARGS[@]}"
  echo "[$(date +%H:%M:%S)] phase814 $ROUND: done $MODEL"
done

python tests/glm5/phase814_tokenizer_valid_answer_unit_closure.py \
  --round-name "$ROUND" \
  --summarize-only
