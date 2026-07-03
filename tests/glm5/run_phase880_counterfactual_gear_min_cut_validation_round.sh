#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-gear_subset_phase879}"
TRANSITION_CLASSES="${2:-clean_causal_transition,nonclean_output_transition}"
ROUTES="${3:-}"
MAX_CANDIDATES="${4:-0}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

COMMON_ARGS=(
  --round-name "$ROUND_NAME"
  --transition-classes "$TRANSITION_CLASSES"
  --routes "$ROUTES"
  --max-candidates "$MAX_CANDIDATES"
  --attn-implementations "flash_attention_2,sdpa"
)

for MODEL in qwen3 glm4 deepseek7b; do
  echo "===== Phase 880 model: ${MODEL} ====="
  python tests/glm5/phase880_counterfactual_gear_min_cut_validation.py --model "$MODEL" "${COMMON_ARGS[@]}"
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

python tests/glm5/phase880_counterfactual_gear_min_cut_validation.py --round-name "$ROUND_NAME" --summarize-round
