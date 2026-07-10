#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROUND="${1:-full_vocabulary_mediation}"
export PYTHONPATH="${ROOT}/tests/gpt5:${PYTHONPATH:-}"

python "${ROOT}/tests/gpt5/phase329_full_vocabulary_case_bank.py" --round "${ROUND}"
for model in qwen3 glm4 deepseek7b; do
  for mechanism in color_retrieval category_retrieval habitat_retrieval; do
    python "${ROOT}/tests/gpt5/phase329_full_vocabulary_mediation.py" \
      --model "${model}" --mechanism "${mechanism}" --round "${ROUND}"
  done
  python "${ROOT}/tests/gpt5/phase329_full_vocabulary_mediation.py" \
    --model "${model}" --collect-model --round "${ROUND}"
done
python "${ROOT}/tests/gpt5/phase329_full_vocabulary_mediation.py" \
  --collect --round "${ROUND}"
