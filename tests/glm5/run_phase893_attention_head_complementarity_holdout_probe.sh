#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ROUND_NAME="${1:-attention_head_complementarity_holdout}"

python tests/glm5/phase893_attention_head_complementarity_holdout_probe.py --model qwen3 --round-name "$ROUND_NAME"
python tests/glm5/phase893_attention_head_complementarity_holdout_probe.py --model glm4 --round-name "$ROUND_NAME"
python tests/glm5/phase893_attention_head_complementarity_holdout_probe.py --model deepseek7b --round-name "$ROUND_NAME"
python tests/glm5/phase893_attention_head_complementarity_holdout_probe.py --round-name "$ROUND_NAME" --summarize-round
