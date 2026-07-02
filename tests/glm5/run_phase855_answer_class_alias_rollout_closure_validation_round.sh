#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-smoke}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

case "$ROUND_NAME" in
  smoke)
    CLOSED=3
    OPEN=2
    STRONG_OTHER=2
    CONTROLS=2
    MAX_SOURCES=0
    TOPK=15
    MAX_NEW=8
    LOG_EVERY=1
    ;;
  main)
    CLOSED=12
    OPEN=6
    STRONG_OTHER=6
    CONTROLS=6
    MAX_SOURCES=0
    TOPK=30
    MAX_NEW=8
    LOG_EVERY=4
    ;;
  confirm)
    CLOSED=32
    OPEN=16
    STRONG_OTHER=16
    CONTROLS=12
    MAX_SOURCES=0
    TOPK=50
    MAX_NEW=10
    LOG_EVERY=6
    ;;
  *)
    echo "unknown round: $ROUND_NAME" >&2
    exit 2
    ;;
esac

MODELS=(qwen3 glm4 deepseek7b)

for MODEL in "${MODELS[@]}"; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase855 ${ROUND_NAME}: ${MODEL}"
  python tests/glm5/phase855_answer_class_alias_rollout_closure_validation.py \
    --model "$MODEL" \
    --round-name "$ROUND_NAME" \
    --source-round confirm \
    --max-strong-target-class-closed "$CLOSED" \
    --max-strong-target-class-open "$OPEN" \
    --max-strong-other "$STRONG_OTHER" \
    --max-controls "$CONTROLS" \
    --max-sources "$MAX_SOURCES" \
    --include-min-cut-conditions \
    --max-min-cut-conditions-per-source 1 \
    --topk-blockers "$TOPK" \
    --max-new-tokens "$MAX_NEW" \
    --log-every "$LOG_EVERY"
done

python tests/glm5/phase855_answer_class_alias_rollout_closure_validation.py \
  --round-name "$ROUND_NAME" \
  --summarize-only
