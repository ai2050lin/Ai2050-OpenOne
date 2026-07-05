#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

ROUND_NAME="${1:-color_feature_neuron_atlas}"
COMMON_ARGS=(
  --round-name "${ROUND_NAME}"
  --templates-per-object 4
  --layers auto
  --batch-size 4
  --topk-blockers 16
  --keep-top-channels-per-sample 128
  --keep-channel-rows 20000
  --summary-top-channels 50
  --log-every 5
)

for MODEL in qwen3 glm4 deepseek7b; do
  python tests/glm5/phase941_color_feature_neuron_atlas.py \
    --model "${MODEL}" \
    "${COMMON_ARGS[@]}"
done

python tests/glm5/phase941_color_feature_neuron_atlas.py \
  --summarize-round \
  --round-name "${ROUND_NAME}"
