#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export TOKENIZERS_PARALLELISM=false
export PROBE_ATTN_IMPLEMENTATION=eager
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python tests/gpt5/phase432_prechoice_terminal_protocol.py >/dev/null
python -m unittest tests.gpt5.test_phase432_prechoice_terminal

# Models are loaded and fully released one at a time.
PROBE_TORCH_DTYPE=float16 python tests/gpt5/phase432_prechoice_terminal_collect.py --model qwen3 --stage open
PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase432_prechoice_terminal_collect.py --model glm4 --stage open
PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase432_prechoice_terminal_collect.py --model deepseek7b --stage open
python tests/gpt5/phase432_prechoice_terminal_analysis.py open

if python - <<'PY'
import json
from pathlib import Path
gate = json.loads(Path("tests/gpt5/result/phase432_prechoice_terminal/phase432_open_gate.json").read_text())
raise SystemExit(0 if gate["sealed_unlock"] else 1)
PY
then
  PROBE_TORCH_DTYPE=float16 python tests/gpt5/phase432_prechoice_terminal_collect.py --model qwen3 --stage sealed
  python tests/gpt5/phase432_prechoice_terminal_analysis.py sealed
fi

python tests/gpt5/phase432_prechoice_terminal_analysis.py failure
python tests/gpt5/phase432_prechoice_terminal_analysis.py summary
python tests/gpt5/phase432_prechoice_terminal_analysis.py publish
python -m unittest tests.gpt5.test_phase432_prechoice_terminal
