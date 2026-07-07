#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-readout_regime_selection_atlas}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${ROOT_DIR}"

for MODEL in qwen3 glm4 deepseek7b; do
  echo "== Phase229 ${MODEL} =="
  python tests/gpt5/phase229_readout_regime_selection_atlas.py \
    --model "${MODEL}" \
    --round-name "${ROUND_NAME}"
done

echo "== Phase229 summarize =="
python tests/gpt5/phase229_readout_regime_selection_atlas.py \
  --summarize \
  --round-name "${ROUND_NAME}"
