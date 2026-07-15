#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export TOKENIZERS_PARALLELISM=false
export PROBE_ATTN_IMPLEMENTATION=eager
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python tests/gpt5/phase431_position_time_protocol.py >/dev/null
python -m unittest tests.gpt5.test_phase431_position_time

# Models are deliberately loaded and released one at a time.
PROBE_TORCH_DTYPE=float16 python tests/gpt5/phase431_position_time_collect.py identity --model qwen3
PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase431_position_time_collect.py identity --model glm4
PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase431_position_time_collect.py identity --model deepseek7b
python tests/gpt5/phase431_position_time_analysis.py identity

PROBE_TORCH_DTYPE=float16 python tests/gpt5/phase431_position_time_collect.py behavior --stage open
python tests/gpt5/phase431_position_time_analysis.py behavior-open
PROBE_TORCH_DTYPE=float16 python tests/gpt5/phase431_position_time_collect.py physical --stage open
python tests/gpt5/phase431_position_time_analysis.py physical-open
python tests/gpt5/phase431_position_time_analysis.py open-gate
python tests/gpt5/phase431_position_time_analysis.py posthoc
python tests/gpt5/phase431_position_time_analysis.py summary

if python - <<'PY'
import json
from pathlib import Path
gate = json.loads(Path("tests/gpt5/result/phase431_position_time/phase431_open_gate.json").read_text())
raise SystemExit(0 if gate["sealed_unlock"] else 1)
PY
then
  PROBE_TORCH_DTYPE=float16 python tests/gpt5/phase431_position_time_collect.py behavior --stage sealed
  PROBE_TORCH_DTYPE=float16 python tests/gpt5/phase431_position_time_collect.py physical --stage sealed
  python tests/gpt5/phase431_position_time_analysis.py sealed
fi

python tests/gpt5/phase431_position_time_analysis.py publish
python -m unittest tests.gpt5.test_phase431_position_time
