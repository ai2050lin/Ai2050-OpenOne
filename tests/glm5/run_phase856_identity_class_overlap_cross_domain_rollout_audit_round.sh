#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-smoke}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

case "$ROUND_NAME" in
  smoke)
    DOMAINS="geometry,animal,tool"
    MAX_PER_DOMAIN=2
    PROMPTS="natural_question"
    TOPK=20
    MAX_NEW=8
    LOG_EVERY=2
    ;;
  main)
    DOMAINS="geometry,animal,tool,color,material,abstract"
    MAX_PER_DOMAIN=4
    PROMPTS="natural_question,natural_category"
    TOPK=30
    MAX_NEW=8
    LOG_EVERY=6
    ;;
  confirm)
    DOMAINS="geometry,animal,tool,color,material,abstract,plant,object"
    MAX_PER_DOMAIN=5
    PROMPTS="natural_question,natural_category,object_only"
    TOPK=50
    MAX_NEW=10
    LOG_EVERY=8
    ;;
  *)
    echo "unknown round: $ROUND_NAME" >&2
    exit 2
    ;;
esac

MODELS=(qwen3 glm4 deepseek7b)

for MODEL in "${MODELS[@]}"; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase856 ${ROUND_NAME}: ${MODEL}"
  python tests/glm5/phase856_identity_class_overlap_cross_domain_rollout_audit.py \
    --model "$MODEL" \
    --round-name "$ROUND_NAME" \
    --domains "$DOMAINS" \
    --max-cases-per-domain "$MAX_PER_DOMAIN" \
    --prompt-variants "$PROMPTS" \
    --topk-tokens "$TOPK" \
    --max-new-tokens "$MAX_NEW" \
    --log-every "$LOG_EVERY"
done

python tests/glm5/phase856_identity_class_overlap_cross_domain_rollout_audit.py \
  --round-name "$ROUND_NAME" \
  --summarize-only
