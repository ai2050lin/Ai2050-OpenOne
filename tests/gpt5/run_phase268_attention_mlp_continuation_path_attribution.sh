#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python tests/gpt5/phase268_attention_mlp_continuation_path_attribution.py \
  --round-name attention_mlp_continuation_path_attribution \
  --cases-per-model 6
