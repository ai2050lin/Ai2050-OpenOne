#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-single_channel_sign_decomposition_atlas}"
SCRIPT="tests/gpt5/phase198_single_channel_sign_decomposition_atlas.py"
RESULT_DIR="tests/result/phase198_single_channel_sign_decomposition_atlas/${ROUND_NAME}"

run_model() {
  local model="$1"
  echo "== Phase198 ${model} =="
  set +e
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
    --boost-factor 1.5 \
    --batch-size 8
  local code="$?"
  set -e
  if [[ "${code}" -ne 0 ]]; then
    echo "Phase198 ${model} exited with code ${code}; checking whether summary was written."
    python - "${RESULT_DIR}/phase198_${model}_summary.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
data = json.loads(path.read_text(encoding="utf-8"))
if data.get("status") != "complete":
    raise SystemExit(1)
print(f"summary complete: {path}")
PY
  fi
}

run_model qwen3
run_model glm4
run_model deepseek7b

python "${SCRIPT}" --round-name "${ROUND_NAME}" --summarize-round
