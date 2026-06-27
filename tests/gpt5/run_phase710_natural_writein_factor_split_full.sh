#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT="results/glm5_phase710_natural_writein_factor_split"
mkdir -p "$OUT"

COMMON=(
  tests/gpt5/phase710_natural_writein_factor_split.py
  --top-heads 32
  --max-new-tokens 8
  --log-every 12
  --hard-exit-after-model
)

python "${COMMON[@]}" --model qwen3 2>&1 | tee "$OUT/phase710_qwen3_run.log"
python "${COMMON[@]}" --model glm4 2>&1 | tee "$OUT/phase710_glm4_run.log"
python "${COMMON[@]}" --model deepseek7b 2>&1 | tee "$OUT/phase710_deepseek7b_run.log"
python tests/gpt5/phase710_natural_writein_factor_split.py --summarize-only 2>&1 | tee "$OUT/phase710_summary_run.log"
