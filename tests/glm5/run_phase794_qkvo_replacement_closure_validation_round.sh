#!/usr/bin/env bash
set -euo pipefail

ROUND="${1:-smoke}"
EXTRA_ARGS=("${@:2}")
SCRIPT="tests/glm5/phase794_qkvo_replacement_closure_validation.py"

case "$ROUND" in
  smoke)
    COMMON_ARGS=(
      --round-name smoke
      --max-cases 1
      --max-routes 2
      --subspace-modes positive
      --budgets 1024
      --attn-source-set-size 4
      --max-components-per-kind 1
      --source-groups all_pre_answer
      --intervention-ops q_answer_replace,o_answer_replace
      --max-new-tokens 2
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
      --source-groups target_value_tokens,instruction,all_pre_answer
      --intervention-ops q_answer_replace,k_source_replace,v_source_replace,o_answer_replace
      --max-new-tokens 3
      --log-every 1
    )
    ;;
  confirm)
    COMMON_ARGS=(
      --round-name confirm
      --max-cases 5
      --max-routes 2
      --subspace-modes positive,negative
      --budgets 1024
      --attn-source-set-size 8
      --max-components-per-kind 1
      --source-groups target_value_tokens,instruction,question,all_pre_answer
      --intervention-ops q_answer_replace,k_source_replace,v_source_replace,o_answer_replace
      --max-new-tokens 3
      --log-every 1
    )
    ;;
  *)
    echo "unknown round: $ROUND" >&2
    exit 2
    ;;
esac

for MODEL in qwen3 glm4 deepseek7b; do
  echo "[$(date +%H:%M:%S)] phase794 ${ROUND}: start ${MODEL}"
  python "$SCRIPT" --model "$MODEL" "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
  echo "[$(date +%H:%M:%S)] phase794 ${ROUND}: done ${MODEL}"
done

python "$SCRIPT" --round-name "$ROUND" --summarize-only
