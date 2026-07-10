#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROUND="${1:-natural_retrieval_path}"
export PYTHONPATH="${ROOT}/tests/gpt5:${PYTHONPATH:-}"

python "${ROOT}/tests/gpt5/phase327_natural_retrieval_case_bank.py" --round "${ROUND}"
for model in qwen3 glm4; do
  python "${ROOT}/tests/gpt5/phase327_natural_retrieval_path.py" --model "${model}" --round "${ROUND}"
done
python "${ROOT}/tests/gpt5/phase327_natural_retrieval_path.py" --model deepseek7b --round "${ROUND}" --stage ab
for start in 0 36 72; do
  end=$((start + 36))
  python "${ROOT}/tests/gpt5/phase327_natural_retrieval_path.py" \
    --model deepseek7b --round "${ROUND}" --stage cd-chunk \
    --case-start "${start}" --case-end "${end}"
done
python "${ROOT}/tests/gpt5/phase327_natural_retrieval_path.py" \
  --model deepseek7b --round "${ROUND}" --stage finalize
python "${ROOT}/tests/gpt5/phase327_natural_retrieval_path.py" --collect --round "${ROUND}"
