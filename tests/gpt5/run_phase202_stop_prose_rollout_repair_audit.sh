#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-stop_prose_rollout_repair_audit}"
SCRIPT="tests/gpt5/phase202_stop_prose_rollout_repair_audit.py"

run_model() {
  local model="$1"
  echo "== Phase202 ${model} =="
  python "${SCRIPT}" \
    --model "${model}" \
    --round-name "${ROUND_NAME}" \
    --phase201-round stop_prose_component_atlas \
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
    --max-candidates 3 \
    --max-samples-per-candidate 16 \
    --prompt-protocols plain,short_answer,stop_explicit \
    --rollout-modes natural,post_answer \
    --boost-factor 1.5 \
    --max-new-tokens 8 \
    --batch-size 8
}

run_model qwen3
run_model glm4
run_model deepseek7b

python "${SCRIPT}" --round-name "${ROUND_NAME}" --summarize-round
