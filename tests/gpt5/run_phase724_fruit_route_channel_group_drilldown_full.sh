#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/rankrank/Documents/OpenOne/Ai2050-OpenOne"
cd "$ROOT"

OUT="results/glm5_phase724_fruit_route_channel_group_drilldown"
mkdir -p "$OUT"

python tests/gpt5/phase724_fruit_route_channel_group_drilldown.py \
  --model qwen3 \
  --hard-exit-after-model 2>&1 | tee "$OUT/qwen3.log"

python tests/gpt5/phase724_fruit_route_channel_group_drilldown.py \
  --model glm4 \
  --hard-exit-after-model 2>&1 | tee "$OUT/glm4.log"

python tests/gpt5/phase724_fruit_route_channel_group_drilldown.py \
  --model deepseek7b \
  --hard-exit-after-model 2>&1 | tee "$OUT/deepseek7b.log"

python tests/gpt5/phase724_fruit_route_channel_group_drilldown.py \
  --summarize-only 2>&1 | tee "$OUT/summary.log"
