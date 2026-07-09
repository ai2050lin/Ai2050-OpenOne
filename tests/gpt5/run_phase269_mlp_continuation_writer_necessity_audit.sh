#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python tests/gpt5/phase269_mlp_continuation_writer_necessity_audit.py \
  --round-name mlp_continuation_writer_necessity_audit \
  --cases-per-model 2 \
  --rollout-tokens 8
