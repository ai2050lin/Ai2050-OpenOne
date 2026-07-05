#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-semantic_direction_coordinate_bridge_audit}"
SCRIPT="tests/glm5/phase941_semantic_direction_coordinate_bridge_audit.py"
RESULT_DIR="tests/result/phase941_semantic_direction_coordinate_bridge_audit/${ROUND_NAME}"

run_model() {
  local model="$1"
  echo "== Phase941 ${model} =="
  set +e
  python "${SCRIPT}" \
    --model "${model}" \
    --round-name "${ROUND_NAME}" \
    --phase939-round bilingual_specificity_tightening_audit \
    --phase940-round semantic_boundary_bridge_audit \
    --min-phase940-bridge-gain 0.02 \
    --max-specs-per-pair 8 \
    --topks "64,256" \
    --templates-per-language 2 \
    --alphas "0.25,0.5,1.0" \
    --batch-size 8 \
    --log-every 12
  local code="$?"
  set -e
  if [[ "${code}" -ne 0 ]]; then
    echo "Phase941 ${model} exited with code ${code}; checking whether summary was written."
    python - "${RESULT_DIR}/phase941_${model}_summary.json" <<'PY'
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
  echo "Phase941 summary exited with code ${summary_code}; checking whether cross summary was written."
  python - "${RESULT_DIR}/phase941_cross_model_summary.json" <<'PY'
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
