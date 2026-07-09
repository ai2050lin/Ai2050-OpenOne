#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ROUND_NAME="${PHASE275_ROUND_NAME:-selected_gap_batch_physical_path_fill}"
MAX_CASES_PER_MODEL="${PHASE275_MAX_CASES_PER_MODEL:-3}"
ROLLOUT_TOKENS="${PHASE275_ROLLOUT_TOKENS:-6}"
ATTN_IMPLEMENTATIONS="${PHASE275_ATTN_IMPLEMENTATIONS:-flash_attention_2,sdpa,eager}"

for MODEL in qwen3 glm4 deepseek7b; do
  echo "[phase275] start ${MODEL} at $(date '+%Y-%m-%d %H:%M:%S')"
  python3 tests/gpt5/phase275_selected_gap_batch_physical_path_fill.py \
    --model "${MODEL}" \
    --round-name "${ROUND_NAME}" \
    --max-cases-per-model "${MAX_CASES_PER_MODEL}" \
    --rollout-tokens "${ROLLOUT_TOKENS}" \
    --attn-implementations "${ATTN_IMPLEMENTATIONS}"
  echo "[phase275] done ${MODEL} at $(date '+%Y-%m-%d %H:%M:%S')"
  python3 - <<'PY'
import gc
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
except Exception:
    pass
gc.collect()
PY
done

python3 tests/gpt5/phase275_selected_gap_batch_physical_path_fill.py \
  --round-name "${ROUND_NAME}" \
  --summarize
