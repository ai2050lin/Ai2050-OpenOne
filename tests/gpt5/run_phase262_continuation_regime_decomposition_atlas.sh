#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python tests/gpt5/phase262_continuation_regime_decomposition_atlas.py \
  --round-name continuation_regime_decomposition_atlas \
  --max-cases-per-mode 8
