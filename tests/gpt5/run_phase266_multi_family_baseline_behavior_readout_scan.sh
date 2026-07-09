#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python tests/gpt5/phase266_multi_family_baseline_behavior_readout_scan.py \
  --round-name multi_family_baseline_behavior_readout_scan \
  --max-cases-per-family 36 \
  --rollout-tokens 12
