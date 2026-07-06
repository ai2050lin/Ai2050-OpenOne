#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-stop_prose_component_atlas}"
SCRIPT="tests/gpt5/phase201_stop_prose_component_atlas.py"

run_model() {
  local model="$1"
  echo "== Phase201 ${model} =="
  python "${SCRIPT}" \
    --model "${model}" \
    --round-name "${ROUND_NAME}" \
    --phase944-round activation_weighted_mlp_channel_causal_audit \
    --phase939-round bilingual_specificity_tightening_audit \
    --phase940-round semantic_boundary_bridge_audit \
    --phase943-round consensus_coordinate_component_mapping_audit \
    --min-phase940-bridge-gain 0.02 \
    --max-specs-per-pair 12 \
    --train-fraction 0.5 \
    --min-train-specs 4 \
    --min-holdout-specs 3 \
    --templates-per-language 2 \
    --max-pairs 4 \
    --max-samples-per-pair 18 \
    --prompt-protocols plain,short_answer,stop_explicit \
    --top-channels-per-layer 4 \
    --max-causal-per-type 3 \
    --max-causal-samples 18 \
    --boost-factor 1.5 \
    --batch-size 8
}

run_model qwen3
run_model glm4
run_model deepseek7b

python "${SCRIPT}" --round-name "${ROUND_NAME}" --summarize-round
