#!/usr/bin/env bash
set -euo pipefail

ROUND="${1:-smoke}"
if [[ $# -gt 0 ]]; then
  shift
fi

SCRIPT="tests/glm5/phase788_matched_source_unit_causal_fiber_validation.py"

case "${ROUND}" in
  smoke)
    MAX_CASES=1
    MAX_ROUTES=1
    BUDGETS="1024"
    MODES="positive,negative"
    ATTN_SET=3
    MLP_SET=8
    MAX_COMPONENTS=1
    TOP_K=5
    LOG_EVERY=1
    ;;
  main)
    MAX_CASES=6
    MAX_ROUTES=2
    BUDGETS="1024"
    MODES="positive,negative"
    ATTN_SET=8
    MLP_SET=32
    MAX_COMPONENTS=2
    TOP_K=10
    LOG_EVERY=2
    ;;
  confirm)
    MAX_CASES=12
    MAX_ROUTES=2
    BUDGETS="1024"
    MODES="positive,negative"
    ATTN_SET=8
    MLP_SET=48
    MAX_COMPONENTS=2
    TOP_K=10
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
      --top-k "${TOP_K}" \
      "$@"
    exit 0
  fi
done

for MODEL in qwen3 glm4 deepseek7b; do
  echo "[$(date '+%H:%M:%S')] phase788 ${ROUND}: start ${MODEL}"
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
    --top-k "${TOP_K}" \
    --log-every "${LOG_EVERY}" \
    "$@"
  echo "[$(date '+%H:%M:%S')] phase788 ${ROUND}: done ${MODEL}"
done

python "${SCRIPT}" --round-name "${ROUND}" --summarize-only "$@"
