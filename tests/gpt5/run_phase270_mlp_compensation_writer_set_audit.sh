#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python tests/gpt5/phase270_mlp_compensation_writer_set_audit.py \
  --round-name mlp_compensation_writer_set_audit \
  --cases-per-model 2 \
  --rollout-tokens 8
