#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT="results/glm5_phase709_natural_generation_writein_closure"
mkdir -p "$OUT"

COMMON=(
  tests/gpt5/phase709_natural_generation_writein_closure.py
  --top-heads 32
  --channel-counts 512
  --max-new-tokens 8
  --log-every 12
  --hard-exit-after-model
)

python "${COMMON[@]}" --model qwen3 2>&1 | tee "$OUT/phase709_qwen3_run.log"
python "${COMMON[@]}" --model glm4 2>&1 | tee "$OUT/phase709_glm4_run.log"
python "${COMMON[@]}" --model deepseek7b 2>&1 | tee "$OUT/phase709_deepseek7b_run.log"
python tests/gpt5/phase709_natural_generation_writein_closure.py --summarize-only 2>&1 | tee "$OUT/phase709_summary_run.log"
