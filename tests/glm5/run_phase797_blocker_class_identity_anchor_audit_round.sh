#!/usr/bin/env bash
set -euo pipefail

ROUND="${1:-confirm}"
EXTRA_ARGS=("${@:2}")
SCRIPT="tests/glm5/phase797_blocker_class_identity_anchor_audit.py"

case "$ROUND" in
  smoke)
    COMMON_ARGS=(
      --round-name smoke
      --max-class-subset-size 3
    )
    ;;
  main)
    COMMON_ARGS=(
      --round-name main
      --max-class-subset-size 4
    )
    ;;
  confirm)
    COMMON_ARGS=(
      --round-name confirm
      --max-class-subset-size 4
    )
    ;;
  *)
    echo "unknown round: $ROUND" >&2
    exit 2
    ;;
esac

echo "[$(date +%H:%M:%S)] phase797 ${ROUND}: start offline blocker-class audit"
python "$SCRIPT" "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
echo "[$(date +%H:%M:%S)] phase797 ${ROUND}: done"
