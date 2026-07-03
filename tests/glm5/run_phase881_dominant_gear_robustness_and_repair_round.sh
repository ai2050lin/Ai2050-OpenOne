#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-dominant_l27c16651_repair}"
MAX_CASES_PER_DOMAIN="${2:-6}"
PROMPT_VARIANTS="${3:-nonclean_direct,semantic_pressure,echo_pressure,format_pressure}"
EDIT_MODES="${4:-flip,half,zero,scale_up}"
MIN_DOMINANT_HITS="${5:-3}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

COMMON_ARGS=(
  --round-name "$ROUND_NAME"
  --max-cases-per-domain "$MAX_CASES_PER_DOMAIN"
  --prompt-variants "$PROMPT_VARIANTS"
  --edit-modes "$EDIT_MODES"
  --min-dominant-hits "$MIN_DOMINANT_HITS"
  --attn-implementations "flash_attention_2,sdpa"
)

for MODEL in qwen3 glm4 deepseek7b; do
  echo "===== Phase 881 model: ${MODEL} ====="
  python tests/glm5/phase881_dominant_gear_robustness_and_repair.py --model "$MODEL" "${COMMON_ARGS[@]}"
  python - <<'PY'
import gc
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
except Exception as exc:
    print(f"cuda cleanup skipped: {exc}")
gc.collect()
PY
done

python tests/glm5/phase881_dominant_gear_robustness_and_repair.py --round-name "$ROUND_NAME" --summarize-round
