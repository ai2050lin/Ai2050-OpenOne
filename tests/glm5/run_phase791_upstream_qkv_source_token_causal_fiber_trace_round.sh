#!/usr/bin/env bash
set -euo pipefail

ROUND="${1:-smoke}"
if [[ $# -gt 0 ]]; then
  shift
fi

SCRIPT="tests/glm5/phase791_upstream_qkv_source_token_causal_fiber_trace.py"

case "${ROUND}" in
  smoke)
    MAX_CASES=1
    MAX_ROUTES=2
    BUDGETS="1024"
    MODES="positive"
    ATTN_SET=3
    MLP_SET=8
    MAX_COMPONENTS=1
    MAX_SOURCE_GROUPS=4
    SOURCE_GROUPS="object_tokens,relation_tokens,answer_prefix,all_pre_answer"
    LOG_EVERY=1
    ;;
  main)
    MAX_CASES=4
    MAX_ROUTES=2
    BUDGETS="1024"
    MODES="positive,negative"
    ATTN_SET=8
    MLP_SET=16
    MAX_COMPONENTS=1
    MAX_SOURCE_GROUPS=6
    SOURCE_GROUPS="object_tokens,relation_tokens,target_value_tokens,candidate_tokens,answer_prefix,all_pre_answer"
    LOG_EVERY=1
    ;;
  confirm)
    MAX_CASES=6
    MAX_ROUTES=2
    BUDGETS="1024"
    MODES="positive,negative"
    ATTN_SET=8
    MLP_SET=16
    MAX_COMPONENTS=1
    MAX_SOURCE_GROUPS=8
    SOURCE_GROUPS="object_tokens,relation_tokens,target_value_tokens,candidate_tokens,answer_prefix,instruction,question,all_pre_answer"
    LOG_EVERY=2
    ;;
  *)
    echo "unknown round: ${ROUND}" >&2
    exit 2
    ;;
esac

for ARG in "$@"; do
  if [[ "${ARG}" == "--dry-run" || "${ARG}" == "--summarize-only" ]]; then
    python "${SCRIPT}" \
      --round-name "${ROUND}" \
      --max-cases "${MAX_CASES}" \
      --route-sizes 6 \
      --max-routes "${MAX_ROUTES}" \
      --budgets "${BUDGETS}" \
      --subspace-modes "${MODES}" \
      --attn-source-set-size "${ATTN_SET}" \
      --mlp-source-set-size "${MLP_SET}" \
      --max-components-per-kind "${MAX_COMPONENTS}" \
      --max-source-groups "${MAX_SOURCE_GROUPS}" \
      --source-groups "${SOURCE_GROUPS}" \
      "$@"
    exit 0
  fi
done

for MODEL in qwen3 glm4 deepseek7b; do
  echo "[$(date '+%H:%M:%S')] phase791 ${ROUND}: start ${MODEL}"
  python "${SCRIPT}" \
    --model "${MODEL}" \
    --round-name "${ROUND}" \
    --max-cases "${MAX_CASES}" \
    --route-sizes 6 \
    --max-routes "${MAX_ROUTES}" \
    --budgets "${BUDGETS}" \
    --subspace-modes "${MODES}" \
    --attn-source-set-size "${ATTN_SET}" \
    --mlp-source-set-size "${MLP_SET}" \
    --max-components-per-kind "${MAX_COMPONENTS}" \
    --max-source-groups "${MAX_SOURCE_GROUPS}" \
    --source-groups "${SOURCE_GROUPS}" \
    --log-every "${LOG_EVERY}" \
    "$@"
  echo "[$(date '+%H:%M:%S')] phase791 ${ROUND}: done ${MODEL}"
done

python "${SCRIPT}" --round-name "${ROUND}" --summarize-only "$@"
