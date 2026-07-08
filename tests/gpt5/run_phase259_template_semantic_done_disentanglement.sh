#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python tests/gpt5/phase259_template_semantic_done_disentanglement.py \
  --round-name template_semantic_done_disentanglement \
  --max-cases-per-mode 8
