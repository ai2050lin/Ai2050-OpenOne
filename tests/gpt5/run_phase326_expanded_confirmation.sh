#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROUND="${1:-distributed_carrier_atlas}"
export PYTHONPATH="${ROOT}/tests/gpt5:${ROOT}/tests/glm5:${PYTHONPATH:-}"

python "${ROOT}/tests/gpt5/phase326_distributed_carrier_case_bank.py" --round "${ROUND}"
for model in qwen3 glm4 deepseek7b; do
  python "${ROOT}/tests/gpt5/phase326_distributed_carrier_atlas.py" --model "${model}" --confirm --round "${ROUND}"
done
python "${ROOT}/tests/gpt5/phase326_distributed_carrier_atlas.py" --collect-confirmation --round "${ROUND}"
