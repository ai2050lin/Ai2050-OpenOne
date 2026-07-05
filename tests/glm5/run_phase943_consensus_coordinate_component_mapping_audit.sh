#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-consensus_coordinate_component_mapping_audit}"
SCRIPT="tests/glm5/phase943_consensus_coordinate_component_mapping_audit.py"
RESULT_DIR="tests/result/phase943_consensus_coordinate_component_mapping_audit/${ROUND_NAME}"

run_model() {
  local model="$1"
  echo "== Phase943 ${model} =="
  set +e
  python "${SCRIPT}" \
    --model "${model}" \
    --round-name "${ROUND_NAME}" \
    --phase939-round bilingual_specificity_tightening_audit \
    --phase940-round semantic_boundary_bridge_audit \
    --min-phase940-bridge-gain 0.02 \
    --max-specs-per-pair 12 \
    --train-fraction 0.5 \
    --min-train-specs 4 \
    --min-holdout-specs 3 \
    --topks "256" \
    --consensus-ks "256" \
    --templates-per-language 2 \
    --max-component-items 12 \
    --batch-size 8
  local code="$?"
  set -e
  if [[ "${code}" -ne 0 ]]; then
    echo "Phase943 ${model} exited with code ${code}; checking whether summary was written."
    python - "${RESULT_DIR}/phase943_${model}_summary.json" <<'PY'
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

set +e
python "${SCRIPT}" --round-name "${ROUND_NAME}" --summarize-round
summary_code="$?"
set -e
if [[ "${summary_code}" -ne 0 ]]; then
  echo "Phase943 summary exited with code ${summary_code}; checking whether cross summary was written."
  python - "${RESULT_DIR}/phase943_cross_model_summary.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
data = json.loads(path.read_text(encoding="utf-8"))
if data.get("status") != "complete":
    raise SystemExit(1)
print(f"cross summary complete: {path}")
PY
fi
