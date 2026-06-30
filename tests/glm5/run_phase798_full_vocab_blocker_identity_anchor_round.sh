#!/usr/bin/env bash
set -euo pipefail

ROUND="${1:-smoke}"
EXTRA_ARGS=("${@:2}")
SCRIPT="tests/glm5/phase798_full_vocab_blocker_identity_anchor.py"

case "$ROUND" in
  smoke)
    COMMON_ARGS=(
      --round-name smoke
      --max-cases 1
      --max-routes 1
      --subspace-modes positive
      --budgets 1024
      --attn-source-set-size 4
      --max-components-per-kind 1
      --source-groups all_pre_answer
      --ladders route_answer,kv_o_route
      --max-route-components 3
      --top-k 40
      --full-rank-window 128
      --max-full-above-classify 20000
      --log-every 1
    )
    ;;
  main)
    COMMON_ARGS=(
      --round-name main
      --max-cases 3
      --max-routes 2
      --subspace-modes positive,negative
      --budgets 1024
      --attn-source-set-size 6
      --max-components-per-kind 1
      --source-groups target_value_tokens,all_pre_answer
      --ladders route_answer,kv_o_route
      --max-route-components 4
      --top-k 64
      --full-rank-window 160
      --max-full-above-classify 30000
      --log-every 1
    )
    ;;
  confirm)
    COMMON_ARGS=(
      --round-name confirm
      --max-cases 6
      --max-routes 2
      --subspace-modes positive,negative
      --budgets 1024
      --attn-source-set-size 8
      --max-components-per-kind 1
      --source-groups target_value_tokens,instruction,all_pre_answer
      --ladders route_answer,kv_o_route
      --max-route-components 4
      --top-k 80
      --full-rank-window 192
      --max-full-above-classify 40000
      --log-every 1
    )
    ;;
  *)
    echo "unknown round: $ROUND" >&2
    exit 2
    ;;
esac

for MODEL in qwen3 glm4 deepseek7b; do
  echo "[$(date +%H:%M:%S)] phase798 ${ROUND}: start ${MODEL}"
  python "$SCRIPT" --model "$MODEL" "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
  echo "[$(date +%H:%M:%S)] phase798 ${ROUND}: done ${MODEL}"
done

python "$SCRIPT" --round-name "$ROUND" --summarize-only
