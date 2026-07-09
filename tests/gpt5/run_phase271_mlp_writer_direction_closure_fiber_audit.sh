#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python tests/gpt5/phase271_mlp_writer_direction_closure_fiber_audit.py \
  --round-name mlp_writer_direction_closure_fiber_audit \
  --cases-per-model 6 \
  --rollout-tokens 6
