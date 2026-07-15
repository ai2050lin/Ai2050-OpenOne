#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python tests/gpt5/phase433_shared_prefix_protocol.py \
  > tests/gpt5/result/phase433_shared_prefix/protocol_stdout.json

PROBE_TORCH_DTYPE=float16 python tests/gpt5/phase433_shared_prefix_collect.py \
  --model qwen3 --stage open --mode all \
  > tests/gpt5/result/phase433_shared_prefix/qwen3_open_collect_stdout.json
PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase433_shared_prefix_collect.py \
  --model glm4 --stage open --mode all \
  > tests/gpt5/result/phase433_shared_prefix/glm4_open_collect_stdout.json
PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase433_shared_prefix_collect.py \
  --model deepseek7b --stage open --mode all \
  > tests/gpt5/result/phase433_shared_prefix/deepseek7b_open_collect_stdout.json

python tests/gpt5/phase433_shared_prefix_analysis.py --stage open \
  > tests/gpt5/result/phase433_shared_prefix/open_analysis_stdout.json

if python - <<'PY'
import json
from pathlib import Path
gate = json.loads(Path("tests/gpt5/result/phase433_shared_prefix/phase433_open_gate.json").read_text())
raise SystemExit(0 if gate.get("sealed_unlock") else 1)
PY
then
  PROBE_TORCH_DTYPE=float16 python tests/gpt5/phase433_shared_prefix_collect.py \
    --model qwen3 --stage sealed --mode all \
    > tests/gpt5/result/phase433_shared_prefix/qwen3_sealed_collect_stdout.json
  python tests/gpt5/phase433_shared_prefix_analysis.py --stage sealed \
    > tests/gpt5/result/phase433_shared_prefix/sealed_analysis_stdout.json
fi

python tests/gpt5/phase433_shared_prefix_analysis.py \
  --stage summary --publish-visual \
  > tests/gpt5/result/phase433_shared_prefix/final_summary_stdout.json
