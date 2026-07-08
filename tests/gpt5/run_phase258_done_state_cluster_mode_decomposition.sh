#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python tests/gpt5/phase258_done_state_cluster_mode_decomposition.py \
  --round-name done_state_cluster_mode_decomposition \
  --max-cases-per-mode 8
