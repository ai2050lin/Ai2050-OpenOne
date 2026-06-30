#!/usr/bin/env bash
set -euo pipefail

ROUND="${1:-smoke}"
if [[ $# -gt 0 ]]; then
  shift
fi

SCRIPT="tests/glm5/phase785_positive_negative_subspace_split.py"

case "${ROUND}" in
  smoke)
    MAX_CASES=1
    MAX_ROUTES=1
    BUDGETS="256"
    MODES="positive,negative,random,all"
    TOP_K=5
    LOG_EVERY=1
    ;;
  main)
    MAX_CASES=6
    MAX_ROUTES=2
    BUDGETS="256,1024"
    MODES="positive,negative,abs,neutral,random,all_positive,all_negative,all"
    TOP_K=10
    LOG_EVERY=2
    ;;
  confirm)
    MAX_CASES=12
    MAX_ROUTES=2
    BUDGETS="256,1024"
    MODES="positive,negative,abs,neutral,random,all_positive,all_negative,all"
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
      "$@"
    exit 0
  fi
done

for MODEL in qwen3 glm4 deepseek7b; do
  echo "[$(date '+%H:%M:%S')] phase785 ${ROUND}: start ${MODEL}"
  python "${SCRIPT}" \
    --model "${MODEL}" \
    --round-name "${ROUND}" \
    --max-cases "${MAX_CASES}" \
    --route-sizes 6 \
    --max-routes "${MAX_ROUTES}" \
    --budgets "${BUDGETS}" \
    --subspace-modes "${MODES}" \
    --top-k "${TOP_K}" \
    --log-every "${LOG_EVERY}" \
    "$@"
  echo "[$(date '+%H:%M:%S')] phase785 ${ROUND}: done ${MODEL}"
done

python "${SCRIPT}" --round-name "${ROUND}" --summarize-only "$@"
