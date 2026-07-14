#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python tests/gpt5/phase421_balanced_boundary_case_bank.py

# Models are deliberately loaded, measured and released one at a time.
python tests/gpt5/phase421_balanced_boundary_behavior.py --model qwen3
PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase421_balanced_boundary_behavior.py --model glm4
python tests/gpt5/phase421_balanced_boundary_behavior.py --model deepseek7b
python tests/gpt5/phase421_balanced_boundary_behavior_analysis.py

AUTHORIZED="$(python -c 'import json; from pathlib import Path; p=Path("tests/gpt5/result/phase421_balanced_boundary_atlas/phase421_physical_development_authorization.json"); print(str(json.loads(p.read_text())["physical_development_collection_authorized"]).lower())')"
if [[ "$AUTHORIZED" != "true" ]]; then
  echo "Phase421 development physical collection was not authorized." >&2
  exit 2
fi

python tests/gpt5/phase421_balanced_boundary_physical.py --model qwen3
PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase421_balanced_boundary_physical.py --model glm4
python tests/gpt5/phase421_balanced_boundary_physical.py --model deepseek7b
python tests/gpt5/phase421_balanced_boundary_physical_analysis.py

python -m unittest tests.gpt5.test_phase421_balanced_boundary_atlas_contract

