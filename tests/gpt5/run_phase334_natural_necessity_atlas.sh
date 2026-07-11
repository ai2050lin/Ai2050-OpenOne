#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python tests/gpt5/phase334_natural_necessity_case_bank.py

for model in qwen3 glm4 deepseek7b; do
  python tests/gpt5/phase334_natural_contrast_survey.py --model "$model" --max-new-tokens 24
  python tests/gpt5/phase334_natural_necessity_intervention.py --model "$model" --stage calibration --max-new-tokens 24
  python tests/gpt5/phase334_natural_necessity_intervention.py --model "$model" --stage heldout --max-new-tokens 24
done

python tests/gpt5/phase334_natural_necessity_analysis.py --collect
python tests/gpt5/phase334_natural_necessity_analysis.py
python tests/gpt5/phase334_publish_natural_necessity_atlas.py
python -m unittest \
  tests.gpt5.test_phase334_natural_necessity_case_bank \
  tests.gpt5.test_phase334_natural_necessity_analysis \
  tests.gpt5.test_phase334_publish_natural_necessity_atlas
