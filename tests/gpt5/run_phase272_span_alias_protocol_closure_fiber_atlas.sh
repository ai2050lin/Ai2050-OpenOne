#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python tests/gpt5/phase272_span_alias_protocol_closure_fiber_atlas.py \
  --round-name span_alias_protocol_closure_fiber_atlas \
  --cases-per-model 6
