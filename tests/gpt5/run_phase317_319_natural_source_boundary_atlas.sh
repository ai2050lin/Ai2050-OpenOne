#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python tests/gpt5/phase317_natural_source_boundary_case_bank.py

for model in qwen3 glm4 deepseek7b; do
  python tests/gpt5/phase318_natural_source_state_transfer.py \
    --model "$model" \
    --round-name natural_source_state_transfer
done
python tests/gpt5/phase318_natural_source_state_transfer.py \
  --round-name natural_source_state_transfer \
  --summarize

for model in qwen3 glm4 deepseek7b; do
  python tests/gpt5/phase319_heldout_component_mediation.py \
    --model "$model" \
    --round-name heldout_component_mediation \
    --rollout-tokens 8
done
python tests/gpt5/phase319_heldout_component_mediation.py \
  --round-name heldout_component_mediation \
  --summarize
