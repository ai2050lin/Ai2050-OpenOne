#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

families=(
  content_knowledge output_protocol reasoning_constraint syntax_structure language_action
  cross_lingual readout_competition state_drift closure
)

for model in qwen3 glm4 deepseek7b; do
  for family in "${families[@]}"; do
    python tests/gpt5/phase330_global_atlas_survey.py \
      --model "$model" --family "$family" --batch-size 8 --max-new-tokens 8 \
      --refresh-rollout
  done
done
