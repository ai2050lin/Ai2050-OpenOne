#!/usr/bin/env bash
set -euo pipefail

CASES_PER_MODEL="${1:-12}"
ROUND_NAME="${2:-internal_semantic_physical_path_probe}"

cd "$(dirname "$0")/../.."

python tests/gpt5/phase305_internal_semantic_physical_path_probe.py --model qwen3 --cases-per-model "${CASES_PER_MODEL}" --round-name "${ROUND_NAME}"
python tests/gpt5/phase305_internal_semantic_physical_path_probe.py --model glm4 --cases-per-model "${CASES_PER_MODEL}" --round-name "${ROUND_NAME}"
python tests/gpt5/phase305_internal_semantic_physical_path_probe.py --model deepseek7b --cases-per-model "${CASES_PER_MODEL}" --round-name "${ROUND_NAME}"
python tests/gpt5/phase305_internal_semantic_physical_path_probe.py --summarize --round-name "${ROUND_NAME}"
