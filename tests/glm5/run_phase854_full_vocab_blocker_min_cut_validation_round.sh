#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-smoke}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

case "$ROUND_NAME" in
  smoke)
    TARGET_FAIL=2
    STRONG_OTHER=1
    CONTROLS=1
    MAX_SOURCE=0
    CANDIDATES=2
    TOPK=15
    LOG_EVERY=1
    ;;
  main)
    TARGET_FAIL=12
    STRONG_OTHER=6
    CONTROLS=6
    MAX_SOURCE=0
    CANDIDATES=2
    TOPK=30
    LOG_EVERY=4
    ;;
  confirm)
    TARGET_FAIL=32
    STRONG_OTHER=16
    CONTROLS=12
    MAX_SOURCE=0
    CANDIDATES=3
    TOPK=50
    LOG_EVERY=6
    ;;
  *)
    echo "unknown round: $ROUND_NAME" >&2
    exit 2
    ;;
esac

MODELS=(qwen3 glm4 deepseek7b)

for MODEL in "${MODELS[@]}"; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase854 ${ROUND_NAME}: ${MODEL}"
  python tests/glm5/phase854_full_vocab_blocker_min_cut_validation.py \
    --model "$MODEL" \
    --round-name "$ROUND_NAME" \
    --source-round confirm \
    --phase851-round confirm \
    --max-target-fail-rows "$TARGET_FAIL" \
    --max-strong-non-target-rows "$STRONG_OTHER" \
    --max-control-rows "$CONTROLS" \
    --max-source-rows "$MAX_SOURCE" \
    --max-candidates-per-row "$CANDIDATES" \
    --topk-blockers "$TOPK" \
    --log-every "$LOG_EVERY"
done

python tests/glm5/phase854_full_vocab_blocker_min_cut_validation.py \
  --round-name "$ROUND_NAME" \
  --summarize-only
