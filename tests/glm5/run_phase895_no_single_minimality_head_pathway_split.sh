#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ROUND_NAME="${1:-minimality_head_pathway_split}"

python tests/glm5/phase895_no_single_minimality_head_pathway_split.py --model qwen3 --round-name "$ROUND_NAME"
python tests/glm5/phase895_no_single_minimality_head_pathway_split.py --model glm4 --round-name "$ROUND_NAME"
python tests/glm5/phase895_no_single_minimality_head_pathway_split.py --model deepseek7b --round-name "$ROUND_NAME"
python tests/glm5/phase895_no_single_minimality_head_pathway_split.py --round-name "$ROUND_NAME" --summarize-round
