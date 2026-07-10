#!/usr/bin/env bash
set -euo pipefail

PAIRS_PER_GROUP="${1:-5}"
ATTRIBUTES="${2:-category,subclass,color,taste,use}"
ROUND_NAME="${3:-pair_three_position_reuse_delta_path_atlas}"

cd "$(dirname "$0")/../.."

python tests/gpt5/phase309_pair_three_position_reuse_delta_path_atlas.py --model qwen3 --pairs-per-group "${PAIRS_PER_GROUP}" --attributes "${ATTRIBUTES}" --round-name "${ROUND_NAME}"
python tests/gpt5/phase309_pair_three_position_reuse_delta_path_atlas.py --model glm4 --pairs-per-group "${PAIRS_PER_GROUP}" --attributes "${ATTRIBUTES}" --round-name "${ROUND_NAME}"
python tests/gpt5/phase309_pair_three_position_reuse_delta_path_atlas.py --model deepseek7b --pairs-per-group "${PAIRS_PER_GROUP}" --attributes "${ATTRIBUTES}" --round-name "${ROUND_NAME}"
python tests/gpt5/phase309_pair_three_position_reuse_delta_path_atlas.py --summarize --round-name "${ROUND_NAME}"
