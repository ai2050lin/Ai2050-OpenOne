#!/usr/bin/env bash
set -euo pipefail

ROUND="${1:-smoke}"
if [[ $# -gt 0 ]]; then
  shift
fi

SCRIPT="tests/glm5/phase786_head_mlp_source_audit.py"

case "${ROUND}" in
  smoke)
    MAX_CASES=1
    MAX_ROUTES=1
    BUDGETS="1024"
    MODES="positive,negative"
    TOP_MLP_CHANNELS=8
    LOG_EVERY=1
    ;;
  main)
    MAX_CASES=6
    MAX_ROUTES=2
    BUDGETS="1024"
    MODES="positive,negative,all_positive,all_negative"
    TOP_MLP_CHANNELS=16
    LOG_EVERY=2
    ;;
  confirm)
    MAX_CASES=12
    MAX_ROUTES=2
    BUDGETS="1024"
    MODES="positive,negative,all_positive,all_negative"
    TOP_MLP_CHANNELS=24
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
      --top-mlp-channels "${TOP_MLP_CHANNELS}" \
      "$@"
    exit 0
  fi
done

for MODEL in qwen3 glm4 deepseek7b; do
  echo "[$(date '+%H:%M:%S')] phase786 ${ROUND}: start ${MODEL}"
  python "${SCRIPT}" \
    --model "${MODEL}" \
    --round-name "${ROUND}" \
    --max-cases "${MAX_CASES}" \
    --route-sizes 6 \
    --max-routes "${MAX_ROUTES}" \
    --budgets "${BUDGETS}" \
    --subspace-modes "${MODES}" \
    --top-mlp-channels "${TOP_MLP_CHANNELS}" \
    --log-every "${LOG_EVERY}" \
    "$@"
  echo "[$(date '+%H:%M:%S')] phase786 ${ROUND}: done ${MODEL}"
done

python "${SCRIPT}" --round-name "${ROUND}" --summarize-only "$@"
