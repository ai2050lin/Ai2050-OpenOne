#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python tests/gpt5/phase255_modelclose_internal_stop_trace.py \
  --round-name modelclose_internal_stop_trace \
  --max-candidates-per-model 1 \
  --max-new-tokens 96
