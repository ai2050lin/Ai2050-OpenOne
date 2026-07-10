#!/usr/bin/env bash
set -euo pipefail

CASES_PER_MODEL="${1:-12}"
ROUND_NAME="${2:-three_position_semantic_path_trace}"

cd "$(dirname "$0")/../.."

python tests/gpt5/phase307_three_position_semantic_path_trace.py --model qwen3 --cases-per-model "${CASES_PER_MODEL}" --round-name "${ROUND_NAME}"
python tests/gpt5/phase307_three_position_semantic_path_trace.py --model glm4 --cases-per-model "${CASES_PER_MODEL}" --round-name "${ROUND_NAME}"
python tests/gpt5/phase307_three_position_semantic_path_trace.py --model deepseek7b --cases-per-model "${CASES_PER_MODEL}" --round-name "${ROUND_NAME}"
python tests/gpt5/phase307_three_position_semantic_path_trace.py --summarize --round-name "${ROUND_NAME}"
