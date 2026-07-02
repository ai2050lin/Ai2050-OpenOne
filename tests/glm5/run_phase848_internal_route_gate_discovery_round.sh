#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-smoke}"
PHASE845_ROUND="${2:-$ROUND_NAME}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

case "${ROUND_NAME}" in
  smoke)
    TOP_GEARS=4
    SPLIT_TYPES="in_sample"
    ;;
  main)
    TOP_GEARS=6
    SPLIT_TYPES="in_sample,object_holdout,prompt_holdout"
    ;;
  confirm)
    TOP_GEARS=6
    SPLIT_TYPES="in_sample,object_holdout,prompt_holdout"
    ;;
  *)
    echo "unknown round: ${ROUND_NAME}" >&2
    exit 2
    ;;
esac

echo "[Phase848] round=${ROUND_NAME} source_phase845=${PHASE845_ROUND} top_gears=${TOP_GEARS} split_types=${SPLIT_TYPES}"

for MODEL in qwen3 glm4 deepseek7b; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 848 ${ROUND_NAME}: start ${MODEL}"
  python tests/glm5/phase848_internal_route_gate_discovery.py \
    --model "${MODEL}" \
    --round-name "${ROUND_NAME}" \
    --phase845-round "${PHASE845_ROUND}" \
    --top-gears "${TOP_GEARS}" \
    --split-types "${SPLIT_TYPES}" \
    --attn-implementations flash_attention_2,sdpa,eager \
    --log-every 1
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 848 ${ROUND_NAME}: done ${MODEL}"
done

python tests/glm5/phase848_internal_route_gate_discovery.py \
  --round-name "${ROUND_NAME}" \
  --summarize-only

