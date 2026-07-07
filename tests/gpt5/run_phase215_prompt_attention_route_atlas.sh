#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-prompt_attention_route_atlas}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

for MODEL in qwen3 glm4 deepseek7b; do
  echo "== Phase215 ${MODEL} =="
  python tests/gpt5/phase215_prompt_attention_route_atlas.py \
    --model "${MODEL}" \
    --round-name "${ROUND_NAME}" \
    --max-per-pattern-group 8
done

python tests/gpt5/phase215_prompt_attention_route_atlas.py \
  --round-name "${ROUND_NAME}" \
  --summarize
