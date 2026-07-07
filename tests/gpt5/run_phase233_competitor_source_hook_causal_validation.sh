#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python tests/gpt5/phase233_competitor_source_hook_causal_validation.py --model qwen3 "$@"
python tests/gpt5/phase233_competitor_source_hook_causal_validation.py --model glm4 "$@"
python tests/gpt5/phase233_competitor_source_hook_causal_validation.py --model deepseek7b "$@"
python tests/gpt5/phase233_competitor_source_hook_causal_validation.py --summarize
