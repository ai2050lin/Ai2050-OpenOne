#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-template_heldout_path_validation}"

cd "$(dirname "$0")/../.."

python tests/gpt5/phase315_template_heldout_path_validation.py --model qwen3 --round-name "${ROUND_NAME}"
python tests/gpt5/phase315_template_heldout_path_validation.py --model glm4 --round-name "${ROUND_NAME}"
python tests/gpt5/phase315_template_heldout_path_validation.py --model deepseek7b --round-name "${ROUND_NAME}"
python tests/gpt5/phase315_template_heldout_path_validation.py --summarize --round-name "${ROUND_NAME}"
