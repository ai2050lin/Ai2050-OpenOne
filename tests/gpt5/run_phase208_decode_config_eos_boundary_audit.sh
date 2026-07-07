#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-decode_config_eos_boundary_audit}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

for MODEL in qwen3 glm4 deepseek7b; do
  echo "== Phase208 ${MODEL} =="
  python tests/gpt5/phase208_decode_config_eos_boundary_audit.py \
    --model "${MODEL}" \
    --round-name "${ROUND_NAME}" \
    --max-prompts 96 \
    --max-steps 32 \
    --batch-size 4
done

python tests/gpt5/phase208_decode_config_eos_boundary_audit.py \
  --round-name "${ROUND_NAME}" \
  --summarize
