#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

families=(
  content_knowledge
  output_protocol
  reasoning_constraint
  syntax_structure
  language_action
  cross_lingual
  readout_competition
  state_drift
  closure
)
models=(qwen3 glm4 deepseek7b)

python tests/gpt5/phase330_nine_family_case_bank.py --round nine_family_global_atlas

for model in "${models[@]}"; do
  for family in "${families[@]}"; do
    python tests/gpt5/phase330_global_atlas_survey.py \
      --model "$model" \
      --family "$family" \
      --round nine_family_global_atlas \
      --batch-size 8 \
      --max-new-tokens 8
  done
done

python tests/gpt5/phase330_component_candidate_scan.py --prepare
for model in "${models[@]}"; do
  python tests/gpt5/phase330_component_candidate_scan.py --model "$model" --batch-size 8
done
python tests/gpt5/phase330_component_candidate_scan.py --collect

python tests/gpt5/phase330_registered_causal_audit.py --register
for model in "${models[@]}"; do
  python tests/gpt5/phase330_registered_causal_audit.py --model "$model" --max-new-tokens 8
done
python tests/gpt5/phase330_registered_causal_audit.py --collect

python tests/gpt5/phase330_global_atlas_analysis.py
python tests/gpt5/phase330_publish_global_atlas.py
