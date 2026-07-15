#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT="tests/gpt5/result/phase437_position_factor"
mkdir -p "$OUT"

python tests/gpt5/phase437_position_factor_protocol.py > "$OUT/protocol_stdout.json"
python tests/gpt5/test_phase437_position_factor.py > "$OUT/test_stdout.log"

run_model() {
  local model="$1"
  local dtype="$2"
  local stage="$3"
  local mode="$4"
  PROBE_TORCH_DTYPE="$dtype" python tests/gpt5/phase437_position_factor_collect.py \
    --model "$model" --stage "$stage" --mode "$mode" \
    > "$OUT/${model}_${stage}_collect_stdout.log"
}

# CUDA models are intentionally loaded and released one at a time.
run_model qwen3 float16 observer behavior
run_model glm4 bfloat16 observer behavior
run_model deepseek7b bfloat16 observer behavior

python tests/gpt5/phase437_position_factor_analysis.py --stage observer \
  > "$OUT/observer_analysis_stdout.json"

for spec in "qwen3 float16" "glm4 bfloat16" "deepseek7b bfloat16"; do
  read -r model dtype <<< "$spec"
  if MODEL="$model" python - <<'PY'
import json
import os
from pathlib import Path

freeze = json.loads(
    Path("tests/gpt5/result/phase437_position_factor/phase437_observer_freeze.json").read_text()
)
qualified = any(
    value["observer_qualified"]
    for value in freeze["models"][os.environ["MODEL"]]["contracts"].values()
)
raise SystemExit(0 if qualified else 1)
PY
  then
    run_model "$model" "$dtype" behavior behavior
  fi
done

python tests/gpt5/phase437_position_factor_analysis.py --stage behavior \
  > "$OUT/behavior_analysis_stdout.json"

for spec in "qwen3 float16" "glm4 bfloat16" "deepseek7b bfloat16"; do
  read -r model dtype <<< "$spec"
  if MODEL="$model" python - <<'PY'
import json
import os
from pathlib import Path

gate = json.loads(
    Path("tests/gpt5/result/phase437_position_factor/phase437_behavior_gate.json").read_text()
)
models = {row["model"] for row in gate["eligible_model_contracts"]}
raise SystemExit(0 if os.environ["MODEL"] in models else 1)
PY
  then
    run_model "$model" "$dtype" physical all
  fi
done

python tests/gpt5/phase437_position_factor_analysis.py --stage open \
  > "$OUT/open_analysis_stdout.json"

for spec in "qwen3 float16" "glm4 bfloat16" "deepseek7b bfloat16"; do
  read -r model dtype <<< "$spec"
  if MODEL="$model" python - <<'PY'
import json
import os
from pathlib import Path

gate = json.loads(
    Path("tests/gpt5/result/phase437_position_factor/phase437_open_gate.json").read_text()
)
models = {row["model"] for row in gate["sealed_authorized_model_contracts"]}
raise SystemExit(0 if os.environ["MODEL"] in models else 1)
PY
  then
    run_model "$model" "$dtype" sealed all
  fi
done

if python - <<'PY'
import json
from pathlib import Path

gate = json.loads(
    Path("tests/gpt5/result/phase437_position_factor/phase437_open_gate.json").read_text()
)
raise SystemExit(0 if gate["sealed_unlock"] else 1)
PY
then
  python tests/gpt5/phase437_position_factor_analysis.py --stage sealed \
    > "$OUT/sealed_analysis_stdout.json"
fi

python tests/gpt5/phase437_position_factor_analysis.py --stage summary --publish-visual \
  > "$OUT/final_summary_stdout.json"
