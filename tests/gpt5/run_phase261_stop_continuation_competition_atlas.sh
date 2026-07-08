#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python tests/gpt5/phase261_stop_continuation_competition_atlas.py \
  --round-name stop_continuation_competition_atlas \
  --max-cases-per-mode 8
