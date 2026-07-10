#!/usr/bin/env bash
set -euo pipefail

LIMIT_PER_MODEL="${1:-0}"
ROUND_NAME="${2:-expanded_mlp_continue_causal_audit}"

cd "$(dirname "$0")/../.."

python tests/gpt5/phase298_expanded_mlp_continue_causal_audit.py \
  --model qwen3 \
  --limit-per-model "${LIMIT_PER_MODEL}" \
  --round-name "${ROUND_NAME}"

python tests/gpt5/phase298_expanded_mlp_continue_causal_audit.py \
  --model glm4 \
  --limit-per-model "${LIMIT_PER_MODEL}" \
  --round-name "${ROUND_NAME}"

python tests/gpt5/phase298_expanded_mlp_continue_causal_audit.py \
  --model deepseek7b \
  --limit-per-model "${LIMIT_PER_MODEL}" \
  --round-name "${ROUND_NAME}"

python tests/gpt5/phase298_expanded_mlp_continue_causal_audit.py \
  --summarize \
  --round-name "${ROUND_NAME}"
