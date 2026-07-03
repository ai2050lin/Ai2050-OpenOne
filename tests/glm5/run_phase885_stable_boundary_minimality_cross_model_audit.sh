#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:-holdout_minimality_cross_model}"
EVAL_DOMAINS="${2:-geometry,animal,tool,color,material,abstract,plant,object}"
MAX_SAME_DOMAIN_CASES="${3:-10}"
MAX_CROSS_CASES_PER_DOMAIN="${4:-1}"
PROMPT_VARIANTS="${5:-natural_question,natural_category,object_only,classification,question_plain,type_of_completion}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

COMMON_ARGS=(
  --round-name "$ROUND_NAME"
  --eval-domains "$EVAL_DOMAINS"
  --max-same-domain-cases "$MAX_SAME_DOMAIN_CASES"
  --max-cross-cases-per-domain "$MAX_CROSS_CASES_PER_DOMAIN"
  --prompt-variants "$PROMPT_VARIANTS"
  --negative-anchor-policy "no_stable"
  --max-negative-anchors "2"
  --controls "same_layer_random,neighbor_channel,opposite_mode"
  --attn-implementations "flash_attention_2,sdpa"
)

for MODEL in qwen3 glm4 deepseek7b; do
  echo "===== Phase 885 model: ${MODEL} ====="
  python tests/glm5/phase885_stable_boundary_minimality_cross_model_audit.py --model "$MODEL" "${COMMON_ARGS[@]}"
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

python tests/glm5/phase885_stable_boundary_minimality_cross_model_audit.py --round-name "$ROUND_NAME" --summarize-round
