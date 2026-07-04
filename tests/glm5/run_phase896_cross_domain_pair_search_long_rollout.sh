#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ROUND_NAME="${1:-cross_domain_pair_search_long_rollout}"

python tests/glm5/phase896_cross_domain_pair_search_long_rollout.py --model qwen3 --round-name "$ROUND_NAME"
python tests/glm5/phase896_cross_domain_pair_search_long_rollout.py --model glm4 --round-name "$ROUND_NAME"
python tests/glm5/phase896_cross_domain_pair_search_long_rollout.py --model deepseek7b --round-name "$ROUND_NAME"
python tests/glm5/phase896_cross_domain_pair_search_long_rollout.py --round-name "$ROUND_NAME" --summarize-round
