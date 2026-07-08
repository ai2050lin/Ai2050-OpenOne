#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python tests/gpt5/phase257_done_signature_cross_sample_reuse.py \
  --round-name done_signature_cross_sample_reuse \
  --max-cases 40
