#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-confirm}"
PHASE849_ROUND="${2:-$ROUND_NAME}"
PHASE850_ROUND="${3:-$ROUND_NAME}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

case "${ROUND_NAME}" in
  smoke|main|confirm)
    ;;
  *)
    echo "unknown round: ${ROUND_NAME}" >&2
    exit 2
    ;;
esac

echo "[Phase851] round=${ROUND_NAME} phase849=${PHASE849_ROUND} phase850=${PHASE850_ROUND}"

python tests/glm5/phase851_global_atlas_schema_orthogonality_audit.py \
  --round-name "${ROUND_NAME}" \
  --phase849-round "${PHASE849_ROUND}" \
  --phase850-round "${PHASE850_ROUND}"
