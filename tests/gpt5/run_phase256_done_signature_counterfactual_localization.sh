#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python tests/gpt5/phase256_done_signature_counterfactual_localization.py \
  --round-name done_signature_counterfactual_localization
