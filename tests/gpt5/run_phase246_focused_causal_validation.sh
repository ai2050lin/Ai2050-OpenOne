#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-focused_causal_validation}"
MAX_TOTAL_CANDIDATES="${MAX_TOTAL_CANDIDATES:-15}"
MAX_ROLLOUT_TOKENS="${MAX_ROLLOUT_TOKENS:-4}"
PERTURB_SCALE="${PERTURB_SCALE:-0.35}"

for MODEL in qwen3 glm4 deepseek7b; do
  python tests/gpt5/phase246_focused_causal_validation.py \
    --model "${MODEL}" \
    --round-name "${ROUND_NAME}" \
    --max-total-candidates "${MAX_TOTAL_CANDIDATES}" \
    --max-rollout-tokens "${MAX_ROLLOUT_TOKENS}" \
    --perturb-scale "${PERTURB_SCALE}"
done

python tests/gpt5/phase246_focused_causal_validation.py \
  --round-name "${ROUND_NAME}" \
  --summarize
