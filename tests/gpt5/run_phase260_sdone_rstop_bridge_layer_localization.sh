#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python tests/gpt5/phase260_sdone_rstop_bridge_layer_localization.py \
  --round-name sdone_rstop_bridge_layer_localization \
  --max-cases-per-mode 8
