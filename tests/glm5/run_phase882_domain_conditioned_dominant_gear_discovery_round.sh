#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-material_color_domain_discovery}"
DISCOVERY_DOMAINS="${2:-material,color}"
EVAL_DOMAINS="${3:-animal,material,color}"
MAX_CASES_PER_DOMAIN="${4:-6}"
MAX_CANDIDATES_PER_DOMAIN="${5:-3}"
LAYERS="${6:-20,21,22,23,24,25,26,27,28,29,30,31}"
EDIT_MODES="${7:-flip,zero}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

COMMON_ARGS=(
  --round-name "$ROUND_NAME"
  --discovery-domains "$DISCOVERY_DOMAINS"
  --eval-domains "$EVAL_DOMAINS"
  --max-cases-per-domain "$MAX_CASES_PER_DOMAIN"
  --max-candidates-per-domain "$MAX_CANDIDATES_PER_DOMAIN"
  --layers "$LAYERS"
  --prompt-variants "nonclean_direct,semantic_pressure,echo_pressure,format_pressure"
  --edit-modes "$EDIT_MODES"
  --controls "same_layer_random"
  --attn-implementations "flash_attention_2,sdpa"
)

for MODEL in qwen3 glm4 deepseek7b; do
  echo "===== Phase 882 model: ${MODEL} ====="
  python tests/glm5/phase882_domain_conditioned_dominant_gear_discovery.py --model "$MODEL" "${COMMON_ARGS[@]}"
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

python tests/glm5/phase882_domain_conditioned_dominant_gear_discovery.py --round-name "$ROUND_NAME" --summarize-round
