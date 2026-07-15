#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT="tests/gpt5/result/phase435_natural_relation"
mkdir -p "$OUT"

python tests/gpt5/phase435_natural_relation_protocol.py > "$OUT/protocol_stdout.json"
python tests/gpt5/test_phase435_natural_relation.py > "$OUT/test_stdout.log"

PROBE_TORCH_DTYPE=float16 python tests/gpt5/phase435_natural_relation_collect.py \
  --model qwen3 --stage interface --mode behavior > "$OUT/qwen3_interface_collect_stdout.log"
PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase435_natural_relation_collect.py \
  --model glm4 --stage interface --mode behavior > "$OUT/glm4_interface_collect_stdout.log"
PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase435_natural_relation_collect.py \
  --model deepseek7b --stage interface --mode behavior > "$OUT/deepseek7b_interface_collect_stdout.log"

python tests/gpt5/phase435_natural_relation_analysis.py --stage interface \
  > "$OUT/interface_analysis_stdout.json"

for model in qwen3 glm4 deepseek7b; do
  if MODEL="$model" python - <<'PY'
import json
import os
from pathlib import Path

freeze = json.loads(
    Path("tests/gpt5/result/phase435_natural_relation/phase435_interface_freeze.json").read_text()
)
raise SystemExit(
    0 if freeze["models"][os.environ["MODEL"]]["calibration_qualified"] else 1
)
PY
  then
    if [[ "$model" == "qwen3" ]]; then
      dtype=float16
    else
      dtype=bfloat16
    fi
    PROBE_TORCH_DTYPE="$dtype" python tests/gpt5/phase435_natural_relation_collect.py \
      --model "$model" --stage behavior --mode behavior > "$OUT/${model}_behavior_collect_stdout.log"
  fi
done

python tests/gpt5/phase435_natural_relation_analysis.py --stage behavior \
  > "$OUT/behavior_analysis_stdout.json"

for model in qwen3 glm4 deepseek7b; do
  if MODEL="$model" python - <<'PY'
import json
import os
from pathlib import Path

gate = json.loads(
    Path("tests/gpt5/result/phase435_natural_relation/phase435_behavior_gate.json").read_text()
)
eligible = {row["model"] for row in gate.get("eligible_model_contracts", [])}
raise SystemExit(0 if os.environ["MODEL"] in eligible else 1)
PY
  then
    if [[ "$model" == "qwen3" ]]; then
      dtype=float16
    else
      dtype=bfloat16
    fi
    PROBE_TORCH_DTYPE="$dtype" python tests/gpt5/phase435_natural_relation_collect.py \
      --model "$model" --stage physical --mode all > "$OUT/${model}_physical_collect_stdout.log"
  fi
done

python tests/gpt5/phase435_natural_relation_analysis.py --stage open \
  > "$OUT/open_analysis_stdout.json"

for model in qwen3 glm4 deepseek7b; do
  if MODEL="$model" python - <<'PY'
import json
import os
from pathlib import Path

gate = json.loads(
    Path("tests/gpt5/result/phase435_natural_relation/phase435_open_gate.json").read_text()
)
authorized = {row["model"] for row in gate.get("sealed_authorized_model_contracts", [])}
raise SystemExit(0 if os.environ["MODEL"] in authorized else 1)
PY
  then
    if [[ "$model" == "qwen3" ]]; then
      dtype=float16
    else
      dtype=bfloat16
    fi
    PROBE_TORCH_DTYPE="$dtype" python tests/gpt5/phase435_natural_relation_collect.py \
      --model "$model" --stage sealed --mode all > "$OUT/${model}_sealed_collect_stdout.log"
  fi
done

if python - <<'PY'
import json
from pathlib import Path

gate = json.loads(
    Path("tests/gpt5/result/phase435_natural_relation/phase435_open_gate.json").read_text()
)
raise SystemExit(0 if gate.get("sealed_unlock") else 1)
PY
then
  python tests/gpt5/phase435_natural_relation_analysis.py --stage sealed \
    > "$OUT/sealed_analysis_stdout.json"
fi

python tests/gpt5/phase435_natural_relation_analysis.py --stage summary --publish-visual \
  > "$OUT/final_summary_stdout.json"
