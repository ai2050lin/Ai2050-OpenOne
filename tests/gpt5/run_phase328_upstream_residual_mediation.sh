#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROUND="${1:-upstream_residual_mediation}"
export PYTHONPATH="${ROOT}/tests/gpt5:${PYTHONPATH:-}"

for model in qwen3 glm4 deepseek7b; do
  python "${ROOT}/tests/gpt5/phase328_upstream_residual_mediation.py" \
    --model "${model}" --round "${ROUND}"
done
python "${ROOT}/tests/gpt5/phase328_upstream_residual_mediation.py" \
  --collect --round "${ROUND}"
