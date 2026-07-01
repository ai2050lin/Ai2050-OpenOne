#!/usr/bin/env bash
set -euo pipefail

ROUND="${1:-smoke}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

case "$ROUND" in
  smoke)
    MAX_CASES=4
    PROMPT_VARIANTS="exact_choices"
    BATCH_SIZE=24
    MAX_SPANS=72
    MAX_NEW=8
    ;;
  main)
    MAX_CASES=12
    PROMPT_VARIANTS="exact_choices,no_choices"
    BATCH_SIZE=24
    MAX_SPANS=96
    MAX_NEW=8
    ;;
  confirm)
    MAX_CASES=20
    PROMPT_VARIANTS="exact_choices,no_choices"
    BATCH_SIZE=32
    MAX_SPANS=128
    MAX_NEW=8
    ;;
  *)
    echo "unknown round: $ROUND" >&2
    exit 2
    ;;
esac

for MODEL in qwen3 glm4 deepseek7b; do
  echo "[$(date +%H:%M:%S)] phase816 $ROUND: start $MODEL cases=$MAX_CASES variants=$PROMPT_VARIANTS"
  python tests/glm5/phase816_multi_token_answer_span_rollout_closure.py \
    --model "$MODEL" \
    --round-name "$ROUND" \
    --max-cases "$MAX_CASES" \
    --prompt-variants "$PROMPT_VARIANTS" \
    --batch-size "$BATCH_SIZE" \
    --max-span-candidates "$MAX_SPANS" \
    --max-new-tokens "$MAX_NEW" \
    --attn-implementations "flash_attention_2,sdpa,eager" \
    --log-every 1
  echo "[$(date +%H:%M:%S)] phase816 $ROUND: done $MODEL"
done

python tests/glm5/phase816_multi_token_answer_span_rollout_closure.py \
  --round-name "$ROUND" \
  --summarize-only
