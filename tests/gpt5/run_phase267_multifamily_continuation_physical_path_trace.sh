#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python tests/gpt5/phase267_multifamily_continuation_physical_path_trace.py \
  --round-name multifamily_continuation_physical_path_trace \
  --cases-per-family 3
