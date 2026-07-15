#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export TOKENIZERS_PARALLELISM=false
export PROBE_ATTN_IMPLEMENTATION=eager

python tests/gpt5/phase429_typed_route_protocol.py --reuse-frozen

PROBE_TORCH_DTYPE=float16 python tests/gpt5/phase429_typed_route_collect.py --model qwen3 --stage observer_instrument
PROBE_TORCH_DTYPE=float16 python tests/gpt5/phase429_typed_route_collect.py --model qwen3 --stage observer
PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase429_typed_route_collect.py --model glm4 --stage observer_instrument
PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase429_typed_route_collect.py --model glm4 --stage observer
PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase429_typed_route_collect.py --model deepseek7b --stage observer_instrument
PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase429_typed_route_collect.py --model deepseek7b --stage observer
python tests/gpt5/phase429_typed_route_analysis.py --stage observer

if python -c 'import json; from pathlib import Path; d=json.loads(Path("tests/gpt5/result/phase429_typed_route/phase429_interface_freeze.json").read_text()); raise SystemExit(0 if d["models"]["qwen3"]["behavior_authorized"] else 1)'
then
  PROBE_TORCH_DTYPE=float16 python tests/gpt5/phase429_typed_route_collect.py --model qwen3 --stage behavior_instrument
  PROBE_TORCH_DTYPE=float16 python tests/gpt5/phase429_typed_route_collect.py --model qwen3 --stage behavior
fi

if python -c 'import json; from pathlib import Path; d=json.loads(Path("tests/gpt5/result/phase429_typed_route/phase429_interface_freeze.json").read_text()); raise SystemExit(0 if d["models"]["glm4"]["behavior_authorized"] else 1)'
then
  PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase429_typed_route_collect.py --model glm4 --stage behavior_instrument
  PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase429_typed_route_collect.py --model glm4 --stage behavior
fi

if python -c 'import json; from pathlib import Path; d=json.loads(Path("tests/gpt5/result/phase429_typed_route/phase429_interface_freeze.json").read_text()); raise SystemExit(0 if d["models"]["deepseek7b"]["behavior_authorized"] else 1)'
then
  PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase429_typed_route_collect.py --model deepseek7b --stage behavior_instrument
  PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase429_typed_route_collect.py --model deepseek7b --stage behavior
fi

python tests/gpt5/phase429_typed_route_analysis.py --stage behavior
python tests/gpt5/phase429_architecture_physical.py --stage freeze

while IFS= read -r model
do
  case "$model" in
    qwen3) dtype=float16 ;;
    glm4|deepseek7b) dtype=bfloat16 ;;
    *) echo "Unexpected Phase429 physical model: $model" >&2; exit 1 ;;
  esac
  PROBE_TORCH_DTYPE="$dtype" python tests/gpt5/phase429_architecture_physical.py --stage open --model "$model"
done < <(python -c 'import json; from pathlib import Path; d=json.loads(Path("tests/gpt5/result/phase429_typed_route/phase429_physical_protocol.json").read_text()); print("\n".join(d["authorized_models"]))')

python tests/gpt5/phase429_architecture_physical.py --stage analyze-open
python tests/gpt5/phase429_posthoc_audit.py
python -m unittest tests/gpt5/test_phase429_typed_route.py tests/gpt5/test_phase429_architecture_physical.py
node tests/gpt5/phase415_multi_route_vis_source_contract.mjs
