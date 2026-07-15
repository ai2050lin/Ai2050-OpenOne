#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

mkdir -p tests/gpt5/result/phase434_binding_timeline

python tests/gpt5/phase434_binding_timeline_protocol.py \
  > tests/gpt5/result/phase434_binding_timeline/protocol_stdout.json

PROBE_TORCH_DTYPE=float16 python tests/gpt5/phase434_binding_timeline_collect.py \
  --model qwen3 --stage behavior --mode behavior \
  > tests/gpt5/result/phase434_binding_timeline/qwen3_behavior_collect_stdout.log
PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase434_binding_timeline_collect.py \
  --model glm4 --stage behavior --mode behavior \
  > tests/gpt5/result/phase434_binding_timeline/glm4_behavior_collect_stdout.log
PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase434_binding_timeline_collect.py \
  --model deepseek7b --stage behavior --mode behavior \
  > tests/gpt5/result/phase434_binding_timeline/deepseek7b_behavior_collect_stdout.log

python tests/gpt5/phase434_binding_timeline_analysis.py --stage behavior \
  > tests/gpt5/result/phase434_binding_timeline/behavior_analysis_stdout.json

for model in qwen3 glm4 deepseek7b; do
  if MODEL="$model" python - <<'PY'
import json
import os
from pathlib import Path

gate = json.loads(
    Path("tests/gpt5/result/phase434_binding_timeline/phase434_behavior_gate.json").read_text()
)
raise SystemExit(0 if os.environ["MODEL"] in gate.get("eligible_models", []) else 1)
PY
  then
    if [[ "$model" == "qwen3" ]]; then
      dtype=float16
    else
      dtype=bfloat16
    fi
    PROBE_TORCH_DTYPE="$dtype" python tests/gpt5/phase434_binding_timeline_collect.py \
      --model "$model" --stage physical --mode all \
      > "tests/gpt5/result/phase434_binding_timeline/${model}_physical_collect_stdout.log"
  fi
done

python tests/gpt5/phase434_binding_timeline_analysis.py --stage open \
  > tests/gpt5/result/phase434_binding_timeline/open_analysis_stdout.json
python tests/gpt5/phase434_binding_timeline_analysis.py --stage summary --publish-visual \
  > tests/gpt5/result/phase434_binding_timeline/final_summary_stdout.json
