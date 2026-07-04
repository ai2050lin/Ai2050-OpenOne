#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ROUND_NAME="${1:-finite_blocker_band_source_localization}"

COMMON_ARGS=(
  --round-name "${ROUND_NAME}"
  --phase899-round domain_axis_rollout_protocol_audit
  --max-rows-per-model 0
  --max-prefix-tokens 5
  --scale-up-factor 2.0
  --layer-stride 1
  --factors 0.5,0.0
  --band-size 32
  --log-every 4
)

for MODEL in qwen3 glm4 deepseek7b; do
  python tests/glm5/phase912_finite_blocker_band_source_localization.py \
    --model "${MODEL}" \
    "${COMMON_ARGS[@]}"
done

python tests/glm5/phase912_finite_blocker_band_source_localization.py \
  --summarize-round \
  --round-name "${ROUND_NAME}"
