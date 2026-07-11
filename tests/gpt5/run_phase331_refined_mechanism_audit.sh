#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python tests/gpt5/phase331_refined_mechanism_case_bank.py
python tests/gpt5/phase331_refined_mechanism_audit.py --model qwen3 --max-new-tokens 12
python tests/gpt5/phase331_refined_mechanism_audit.py --model glm4 --max-new-tokens 12
python tests/gpt5/phase331_refined_mechanism_audit.py --model deepseek7b --max-new-tokens 12
python tests/gpt5/phase331_refined_mechanism_audit.py --collect
python tests/gpt5/phase331_refined_mechanism_analysis.py
python tests/gpt5/phase331_publish_refined_atlas.py
python -m unittest \
  tests.gpt5.test_phase331_refined_mechanism_case_bank \
  tests.gpt5.test_phase331_refined_mechanism_analysis \
  tests.gpt5.test_phase331_publish_refined_atlas
npm --prefix frontend run build
