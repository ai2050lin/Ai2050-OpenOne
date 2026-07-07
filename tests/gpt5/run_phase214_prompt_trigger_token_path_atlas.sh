#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-prompt_trigger_token_path_atlas}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

for MODEL in qwen3 glm4 deepseek7b; do
  echo "== Phase214 ${MODEL} =="
  python tests/gpt5/phase214_prompt_trigger_token_path_atlas.py \
    --model "${MODEL}" \
    --round-name "${ROUND_NAME}" \
    --max-rows-per-pattern 30
done

python tests/gpt5/phase214_prompt_trigger_token_path_atlas.py \
  --round-name "${ROUND_NAME}" \
  --summarize
