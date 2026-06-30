#!/usr/bin/env bash
set -euo pipefail

ROUND="${1:-smoke}"
EXTRA_ARGS=("${@:2}")
SCRIPT="tests/glm5/phase793_qkvo_independent_causal_decomposition.py"

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
      --source-groups candidate_tokens,all_pre_answer
      --intervention-ops q_answer_zero,k_source_zero,v_source_zero,o_answer_zero
      --log-every 1
    )
    ;;
  main)
    COMMON_ARGS=(
      --round-name main
      --max-cases 4
      --max-routes 2
      --subspace-modes positive,negative
      --budgets 1024
      --attn-source-set-size 6
      --max-components-per-kind 1
      --source-groups candidate_tokens,target_value_tokens,instruction,all_pre_answer
      --intervention-ops q_answer_zero,k_source_zero,v_source_zero,o_answer_zero
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
      --source-groups candidate_tokens,target_value_tokens,instruction,question,all_pre_answer
      --intervention-ops q_answer_zero,k_source_zero,v_source_zero,o_answer_zero
      --log-every 2
    )
    ;;
  *)
    echo "unknown round: $ROUND" >&2
    exit 2
    ;;
esac

for MODEL in qwen3 glm4 deepseek7b; do
  echo "[$(date +%H:%M:%S)] phase793 ${ROUND}: start ${MODEL}"
  python "$SCRIPT" --model "$MODEL" "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
  echo "[$(date +%H:%M:%S)] phase793 ${ROUND}: done ${MODEL}"
done

python "$SCRIPT" --round-name "$ROUND" --summarize-only
