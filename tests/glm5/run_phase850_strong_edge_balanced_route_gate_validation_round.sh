#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-smoke}"
PHASE849_ROUND="${2:-$ROUND_NAME}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

case "${ROUND_NAME}" in
  smoke)
    SPLIT_TYPES="in_sample"
    ;;
  main)
    SPLIT_TYPES="in_sample,object_holdout,prompt_holdout"
    ;;
  confirm)
    SPLIT_TYPES="in_sample,object_holdout,prompt_holdout"
    ;;
  *)
    echo "unknown round: ${ROUND_NAME}" >&2
    exit 2
    ;;
esac

echo "[Phase850] round=${ROUND_NAME} source_phase849=${PHASE849_ROUND} split_types=${SPLIT_TYPES}"

for MODEL in qwen3 glm4 deepseek7b; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 850 ${ROUND_NAME}: start ${MODEL}"
  python tests/glm5/phase850_strong_edge_balanced_route_gate_validation.py \
    --model "${MODEL}" \
    --round-name "${ROUND_NAME}" \
    --phase849-round "${PHASE849_ROUND}" \
    --split-types "${SPLIT_TYPES}" \
    --log-every 1
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 850 ${ROUND_NAME}: done ${MODEL}"
done

python tests/glm5/phase850_strong_edge_balanced_route_gate_validation.py \
  --round-name "${ROUND_NAME}" \
  --summarize-only
