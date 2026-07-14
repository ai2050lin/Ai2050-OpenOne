#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python tests/gpt5/phase416_dual_track_case_bank.py

# A return code of 2 means the old bundled collector gate failed.  Phase416
# intentionally analyzes prefill, cache, and generation as separate domains.
for model in qwen3 glm4 deepseek7b; do
  set +e
  python tests/gpt5/phase416_real_collector_qualification.py --model "$model"
  rc=$?
  set -e
  if [[ "$rc" -ne 0 && "$rc" -ne 2 ]]; then
    exit "$rc"
  fi
done

python tests/gpt5/phase416_qualification_analysis.py

for model in qwen3 glm4 deepseek7b; do
  python tests/gpt5/phase416_prefill_physical_trace.py --model "$model"
done

python tests/gpt5/phase416_prefill_physical_analysis.py
python -m unittest tests/gpt5/test_phase416_formal_world_physical_atlas.py
