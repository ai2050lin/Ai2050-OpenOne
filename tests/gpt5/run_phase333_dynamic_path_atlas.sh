#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python tests/gpt5/phase333_dynamic_case_bank.py
python tests/gpt5/phase333_dynamic_survey.py --model qwen3 --max-new-tokens 64
python tests/gpt5/phase333_residual_block_exchange.py --model qwen3 --max-new-tokens 64
python tests/gpt5/phase333_dynamic_survey.py --model glm4 --max-new-tokens 64
python tests/gpt5/phase333_residual_block_exchange.py --model glm4 --max-new-tokens 64
python tests/gpt5/phase333_dynamic_survey.py --model deepseek7b --max-new-tokens 64
python tests/gpt5/phase333_residual_block_exchange.py --model deepseek7b --max-new-tokens 64
python tests/gpt5/phase333_residual_block_exchange.py --collect
python tests/gpt5/phase333_dynamic_analysis.py
python tests/gpt5/phase333_publish_dynamic_atlas.py
python -m unittest \
  tests.gpt5.test_phase333_dynamic_case_bank \
  tests.gpt5.test_phase333_dynamic_analysis \
  tests.gpt5.test_phase333_publish_dynamic_atlas
