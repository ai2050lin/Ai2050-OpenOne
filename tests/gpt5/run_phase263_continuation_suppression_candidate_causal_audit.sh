#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python tests/gpt5/phase263_continuation_suppression_candidate_causal_audit.py \
  --round-name continuation_suppression_candidate_causal_audit \
  --max-cases-per-mode 8 \
  --max-candidates-per-policy 8 \
  --lambdas 2,4,8,12 \
  --alpha-stop 4 \
  --rollout-candidates 5 \
  --rollout-tokens 24 \
  --rollout-lambda 8
