#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python tests/gpt5/phase332_interface_branch_case_bank.py
python tests/gpt5/phase332_interface_branch_survey.py --model qwen3 --batch-size 4 --max-new-tokens 64
python tests/gpt5/phase332_interface_path_exchange.py --model qwen3 --max-new-tokens 64
python tests/gpt5/phase332_interface_branch_survey.py --model glm4 --batch-size 4 --max-new-tokens 64
python tests/gpt5/phase332_interface_path_exchange.py --model glm4 --max-new-tokens 64
python tests/gpt5/phase332_interface_branch_survey.py --model deepseek7b --batch-size 4 --max-new-tokens 64
python tests/gpt5/phase332_interface_path_exchange.py --model deepseek7b --max-new-tokens 64
python tests/gpt5/phase332_interface_path_exchange.py --collect
python tests/gpt5/phase332_interface_branch_analysis.py
python tests/gpt5/phase332_publish_interface_atlas.py
python -m unittest \
  tests.gpt5.test_phase332_interface_branch_case_bank \
  tests.gpt5.test_phase332_interface_branch_analysis \
  tests.gpt5.test_phase332_publish_interface_atlas
